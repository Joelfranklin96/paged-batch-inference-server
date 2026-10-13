# Paged-Attention Batch Inference Server

A dependency-free (NumPy-only) implementation of the machine that sits between a trained
transformer and a production API — the same ideas behind vLLM's PagedAttention and continuous
batching, rebuilt from first principles so that every allocation, eviction, and scheduling decision
is explicit and readable instead of hidden behind a CUDA kernel.

## Features

- **Paged KV cache** — a fixed-size block pool with per-sequence block tables, so a sequence's keys
  and values can live in non-adjacent blocks and memory tracks tokens actually generated.
- **Continuous batching** — new requests join the running batch mid-flight instead of waiting for
  the current batch to drain, with blocks reclaimed the moment a sequence finishes.
- **Priority scheduler** — capacity-aware admission control, plus preemption that frees a victim's
  blocks and requeues it at its original priority rather than failing it back to the client.
- **Static batching** — a lockstep prefill-then-decode path, kept alongside continuous batching as
  the baseline it improves on.
- **Full sampling stack** — temperature scaling, top-k, top-p (nucleus), and greedy selection,
  routed through a single `select_next_token` path shared by every decode loop.
- **Streaming serving API** — request submission, a driver loop, per-token stream chunks, and
  decoded completion responses with finish reasons.
- **Character-level tokenizer** — vocabulary construction plus encode/decode with `<pad>`, `<bos>`,
  `<eos>`, and `<unk>` handling.
- **Benchmark harness** — time to first token, inter-token latency, token and request throughput,
  and latency percentiles measured through the real serving path.

## Why paged KV cache

Modern LLM serving throughput is won or lost in the serving layer, not the model. A contiguous KV
cache must reserve `max_seq_len` slots per sequence up front, so a request that generates 20 tokens
out of a 2048-token reservation wastes 99% of its memory. The paged allocator here hands out
fixed-size blocks on demand and tracks them in a per-sequence block table, so memory is consumed in
proportion to tokens actually generated, and freed blocks return immediately to a shared pool for
the next admitted request.

Paging also removes both classic forms of fragmentation. **Internal fragmentation** — space wasted
inside a reservation — shrinks from the entire unused tail of a `max_seq_len` reservation to at most
`block_size - 1` unused slots in a sequence's final block. **External fragmentation** — free memory
that exists but can't be used because it isn't contiguous — disappears outright: every block is the
same size and a sequence's keys and values may be scattered anywhere in the pool, because
`gather_kv_from_blocks` reassembles them through the block table instead of relying on adjacency.

```
   paged KV pool     +-------+-------+-------+-------+          seq_tables
   (num_blocks x     | blk 0 | blk 1 | blk 2 | blk 3 | ...      A -> [0, 2]
    block_size x     +-------+-------+-------+-------+          B -> [1, 3]
    d_model)             A       B       A       B
```

That, in turn, is what makes continuous batching possible: the scheduler can admit a new request
mid-flight whenever `has_free_capacity` says the pool can cover its prompt, instead of waiting for
the whole batch to finish.

## Request lifecycle

```
     submit_request
           |
           v
   +----------------+                          +-----------------+
   |  waiting heap  |  --- admission ------->  |  running batch  |
   |   (priority)   |                          | (<= max_running)|
   |                |  <-- preemption -------  |                 |
   +----------------+                          +-----------------+
                                                        |
      schedule_step performs both transitions:          |  continuous_batch_step
      admits while has_free_capacity allows, and        |  decodes one token per
      preempts whenever the running batch exceeds       |  active sequence
      max_running -- freeing the victim's blocks        v
      and pushing it back onto the waiting heap         stream chunk per token
      at its original priority                          |
                                                        v
                                                        completion response
                                                        (sequence's blocks freed)
```

Preemption is non-destructive to the request: an evicted sequence returns to the waiting heap at
its original priority rather than being failed back to the client.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

`demo.py` runs the full stack against a small randomly initialised model — sampling primitives,
tokenizer round-trip, single-sequence generation, static batching, continuous batching, the
streaming server, and the benchmark. Sample output:

```
[sampling] greedy=4 sampled=4 probs=[0.2227 0.     0.     0.     0.7773]
[tokenizer] vocab_size=31 ids=[1, 12, 9, 16, 16, 19, 4, 27, 19, 22, 16, 8] roundtrip='hello world'
[single] ids=[13, 7, 12, 13, 12, 7] text='ichihc'
[static] static-0: text='iwswi'
[continuous] cont-1: text='blobl'
[continuous] allocator={'used': 0, 'free': 32, 'total': 32}
[serving] submitted=['req-0', 'req-1', 'req-2', 'req-3'] streamed_chunks=20
[serving] req-0: tokens=[4, 4, 23, 13, 27] reason=length text='  siw'
[bench] total_time=0.0010s tokens/s=11740.5 requests/s=2935.1
[bench] ttft_p50=0.000385s ttft_p90=0.000388s
```

The weights are random, so the generated text is gibberish by design — the point of the project is
the serving machinery around the model, which is exercised end to end.

## Using it as a library

```python
import numpy as np
from inference_server import (build_vocab, init_model_params, init_block_allocator,
                              submit_request, drive_until_complete, build_completion_response)

rng = np.random.default_rng(0)
vocab = build_vocab(["hello world"], ["<pad>", "<bos>", "<eos>", "<unk>"])
params = init_model_params(len(vocab['token_to_id']), d_model=16, max_seq_len=64, rng=rng)
allocator = init_block_allocator(num_blocks=32, block_size=8, d_model=16)

server_state = {'waiting_heap': [], 'running': [], 'completed': {}, 'streams': {},
                'next_request_id': 0, 'eos_token_id': vocab['token_to_id']['<eos>'],
                'max_running': 4}
sampling_config = {'temperature': 0.8, 'top_k': 5, 'top_p': 0.9, 'rng': rng}

request_id = submit_request(server_state, "hello", max_new_tokens=8, priority=0, vocab=vocab)
chunks = drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps=64)
print(build_completion_response(server_state, request_id, vocab))
```

Every public function is re-exported from the package root, or you can import from the layer you
want directly — `from inference_server.paged_cache import init_block_allocator`.

## Layout

The package is layered strictly bottom-up — each module imports only from the ones above it, so
there are no circular dependencies and any layer can be read or tested on its own.

```
inference_server/
├── sampling.py      softmax, temperature, top-k/top-p, greedy and stochastic selection
├── tokenizer.py     vocabulary construction, encode, decode
├── transformer.py   weight init, embeddings, projections, KV cache, causal attention, prefill/decode
├── paged_cache.py   block allocator, block tables, paged append/gather, paged attention
├── batching.py      sequence state, single / static-batch / continuous-batch decode loops
├── scheduler.py     priority queue, admission control, preemption
├── server.py        request submission, driver loop, stream chunks, completion responses
├── metrics.py       TTFT, inter-token latency, throughput, percentiles
└── benchmark.py     end-to-end benchmark runner
demo.py              runnable tour of every layer
```

## Limitations

A reference implementation, not a production server: single-head, single-layer attention on CPU
NumPy, with no prefix sharing or quantization, and no HTTP layer — `drive_until_complete` is the
in-process loop that a web framework such as FastAPI would call to serve requests over HTTP.
