"""Non-toy benchmark: continuous batching (paged KV) vs static batching.

Config: d_model=256, vocab=4096, max_seq_len=512, block_size=16, num_blocks=256,
8 concurrent requests, 128-token prompts, 64 new tokens each, single-threaded NumPy.

Run:  python bench.py
Writes bench_results.json with raw per-trial numbers and prints a summary.
"""

import os

# Pin BLAS threading before NumPy is imported so "single-threaded" is actually true.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import platform
import statistics
import time

import numpy as np

from inference_server import (
    append_to_paged_cache,
    build_vocab,
    continuous_batch_step,
    drive_until_complete,
    encode_prompt,
    init_block_allocator,
    init_model_params,
    init_sequence_state,
    kv_blocks_in_use,
    make_request,
    model_prefill,
    sequence_decode_step,
    static_batch_generate,
    submit_request,
)
from inference_server.metrics import (
    aggregate_throughput,
    inter_token_latency,
    latency_percentiles,
    time_to_first_token,
)
from inference_server.transformer import embed_tokens, linear_projection

D_MODEL = 256
VOCAB_SIZE = 4096
MAX_SEQ_LEN = 512
BLOCK_SIZE = 16
NUM_BLOCKS = 256
NUM_REQUESTS = 8
PROMPT_TOKENS = 128
MAX_NEW_TOKENS = 64
TRIALS = 5

CHARS = "abcdefghijklmnopqrstuvwxyz 0123456789"


def padded_vocab():
    """Character vocab padded with dummy tokens so the embedding/output matrices are 4096 wide."""
    vocab = build_vocab([CHARS], ["<pad>", "<bos>", "<eos>", "<unk>"])
    while len(vocab['id_to_token']) < VOCAB_SIZE:
        token = "<x{}>".format(len(vocab['id_to_token']))
        vocab['token_to_id'][token] = len(vocab['id_to_token'])
        vocab['id_to_token'].append(token)
    return vocab


def make_prompts(rng):
    # encode_prompt prepends <bos>, so 127 chars -> 128 tokens.
    return ["".join(rng.choice(list(CHARS), size=PROMPT_TOKENS - 1))
            for _ in range(NUM_REQUESTS)]


def sampling_config_for(rng):
    # eos_token_id=-1 disables early stop so every run generates exactly the same token count.
    return {'temperature': 1.0, 'top_k': 5, 'top_p': 0.9, 'eos_token_id': -1, 'rng': rng}


def bench_continuous(params, vocab, prompts, sampling_config):
    """Drive the full server path (scheduler + paged cache + streaming) step by step."""
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    state = {'running': [], 'completed': {}, 'streams': {}, 'waiting_heap': [],
             'next_request_id': 0, 'max_running': NUM_REQUESTS, 'eos_token_id': -1}

    events = []
    peak_blocks = 0
    max_waste_per_seq = 0.0

    start = time.perf_counter()
    for priority, prompt in enumerate(prompts):
        rid = submit_request(state, prompt, MAX_NEW_TOKENS, priority, vocab)
        events.append({'request_id': rid, 'event': 'submit', 'type': 'submit',
                       'time': time.perf_counter() - start})

    while state['waiting_heap'] or state['running']:
        chunks = drive_until_complete(state, params, allocator, sampling_config,
                                      vocab, max_steps=1)
        now = time.perf_counter() - start
        for chunk in chunks:
            events.append({'request_id': chunk['request_id'], 'event': 'token',
                           'type': 'token', 'time': now})
            if chunk['finished']:
                events.append({'request_id': chunk['request_id'], 'event': 'finish',
                               'type': 'finish', 'time': now})

        usage = kv_blocks_in_use(allocator)
        peak_blocks = max(peak_blocks, usage['used'])
        seq_lengths = allocator.get('seq_lengths', {})
        if seq_lengths:
            waste = (usage['used'] * BLOCK_SIZE - sum(seq_lengths.values())) / len(seq_lengths)
            max_waste_per_seq = max(max_waste_per_seq, waste)

    total = time.perf_counter() - start
    ttft = time_to_first_token(events)
    itl = inter_token_latency(events)

    return {
        'total_time_s': total,
        'throughput': aggregate_throughput(events, total),
        'ttft_first_s': min(ttft.values()),
        'ttft_percentiles_s': latency_percentiles(list(ttft.values()), [50, 90, 99]),
        'mean_itl_s': float(np.mean(list(itl.values()))),
        'peak_blocks_used': peak_blocks,
        'max_waste_tokens_per_seq': max_waste_per_seq,
    }


def bench_static(params, vocab, prompts, sampling_config, rng):
    """Lockstep static batching over the same prompts with contiguous per-sequence caches."""
    requests = [make_request("static-{}".format(i), encode_prompt(p, vocab),
                             MAX_NEW_TOKENS, sampling_config)
                for i, p in enumerate(prompts)]
    start = time.perf_counter()
    outputs = static_batch_generate(params, requests, sampling_config,
                                    MAX_NEW_TOKENS, rng=rng)
    total = time.perf_counter() - start
    tokens = sum(len(o['output_ids']) for o in outputs)
    return {'total_time_s': total, 'tokens_per_second': tokens / total,
            'total_tokens': tokens}


def bench_single_sequence(params, vocab, prompt, sampling_config, rng):
    """Per-token decode latency for one sequence on the contiguous and paged cache paths."""
    prompt_ids = encode_prompt(prompt, vocab)

    # Contiguous path.
    request = make_request("solo", prompt_ids, MAX_NEW_TOKENS, sampling_config)
    state = init_sequence_state(request, params)
    contiguous = []
    for _ in range(MAX_NEW_TOKENS):
        t0 = time.perf_counter()
        sequence_decode_step(state, params, rng)
        contiguous.append(time.perf_counter() - t0)

    # Paged path: prefill K/V for the prompt, then step one token at a time.
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    embeddings = embed_tokens(prompt_ids[:-1], params['embedding'])
    append_to_paged_cache(allocator, "solo-paged",
                          linear_projection(embeddings, params['Wk']),
                          linear_projection(embeddings, params['Wv']))
    sequence = {'request_id': "solo-paged", 'token_ids': list(prompt_ids),
                'generated': [], 'length': len(prompt_ids), 'done': False,
                'max_new_tokens': MAX_NEW_TOKENS}
    paged = []
    for _ in range(MAX_NEW_TOKENS):
        t0 = time.perf_counter()
        continuous_batch_step(params, [sequence], allocator, sampling_config)
        paged.append(time.perf_counter() - t0)

    return {
        'contiguous_mean_ms': float(np.mean(contiguous) * 1000),
        'contiguous_median_ms': float(np.median(contiguous) * 1000),
        'paged_mean_ms': float(np.mean(paged) * 1000),
        'paged_median_ms': float(np.median(paged) * 1000),
    }


def capacity_test():
    """Fill the real allocator with full-budget sequences until it reports out-of-memory."""
    allocator = init_block_allocator(NUM_BLOCKS, BLOCK_SIZE, D_MODEL)
    tokens_per_seq = PROMPT_TOKENS + MAX_NEW_TOKENS
    kv = np.zeros((tokens_per_seq, D_MODEL), dtype=np.float32)

    admitted = 0
    try:
        while True:
            append_to_paged_cache(allocator, "cap-{}".format(admitted), kv, kv)
            admitted += 1
    except RuntimeError:
        pass

    # A contiguous cache must reserve max_seq_len slots per sequence from the same pool.
    contiguous = (NUM_BLOCKS * BLOCK_SIZE) // MAX_SEQ_LEN
    return {'paged_concurrent_seqs': admitted, 'contiguous_concurrent_seqs': contiguous,
            'ratio': admitted / contiguous, 'tokens_per_seq': tokens_per_seq,
            'total_kv_slots': NUM_BLOCKS * BLOCK_SIZE}


def main():
    setup_rng = np.random.default_rng(0)
    vocab = padded_vocab()
    params = init_model_params(VOCAB_SIZE, D_MODEL, MAX_SEQ_LEN, setup_rng)
    prompts = make_prompts(setup_rng)

    # Warm-up (excluded from results).
    warm_rng = np.random.default_rng(99)
    bench_continuous(params, vocab, prompts, sampling_config_for(warm_rng))
    bench_static(params, vocab, prompts, sampling_config_for(warm_rng), warm_rng)

    results = {
        'config': {'d_model': D_MODEL, 'vocab_size': VOCAB_SIZE, 'max_seq_len': MAX_SEQ_LEN,
                   'block_size': BLOCK_SIZE, 'num_blocks': NUM_BLOCKS,
                   'num_requests': NUM_REQUESTS, 'prompt_tokens': PROMPT_TOKENS,
                   'max_new_tokens': MAX_NEW_TOKENS, 'trials': TRIALS},
        'env': {'machine': platform.machine(), 'processor': platform.processor(),
                'platform': platform.platform(), 'python': platform.python_version(),
                'numpy': np.__version__,
                'blas_threads': os.environ.get('OMP_NUM_THREADS')},
        'continuous': [], 'static': [], 'single_sequence': [],
    }

    for trial in range(TRIALS):
        trial_rng = np.random.default_rng(1000 + trial)
        results['continuous'].append(
            bench_continuous(params, vocab, prompts, sampling_config_for(trial_rng)))
        results['static'].append(
            bench_static(params, vocab, prompts, sampling_config_for(trial_rng), trial_rng))
        results['single_sequence'].append(
            bench_single_sequence(params, vocab, prompts[0],
                                  sampling_config_for(trial_rng), trial_rng))

    results['capacity'] = capacity_test()

    def med(values):
        return statistics.median(values)

    cont_tps = [r['throughput']['tokens_per_second'] for r in results['continuous']]
    stat_tps = [r['tokens_per_second'] for r in results['static']]
    ttft_first = [r['ttft_first_s'] for r in results['continuous']]
    ttft_p50 = [r['ttft_percentiles_s'][50.0] for r in results['continuous']]
    ttft_p90 = [r['ttft_percentiles_s'][90.0] for r in results['continuous']]
    itl = [r['mean_itl_s'] for r in results['continuous']]
    solo_paged = [r['paged_median_ms'] for r in results['single_sequence']]
    solo_contig = [r['contiguous_median_ms'] for r in results['single_sequence']]
    static_total = [r['total_time_s'] for r in results['static']]

    results['summary'] = {
        'continuous_tokens_per_sec_median': med(cont_tps),
        'continuous_tokens_per_sec_range': [min(cont_tps), max(cont_tps)],
        'static_tokens_per_sec_median': med(stat_tps),
        'throughput_ratio_continuous_vs_static': med(cont_tps) / med(stat_tps),
        'ttft_first_ms_median': med(ttft_first) * 1000,
        'ttft_p50_ms_median': med(ttft_p50) * 1000,
        'ttft_p90_ms_median': med(ttft_p90) * 1000,
        'mean_itl_ms_median': med(itl) * 1000,
        'single_seq_paged_median_ms': med(solo_paged),
        'single_seq_contiguous_median_ms': med(solo_contig),
        'static_total_time_ms_median': med(static_total) * 1000,
        'first_output_ratio_static_vs_continuous_p90':
            med(static_total) * 1000 / (med(ttft_p90) * 1000),
        'peak_blocks_used': max(r['peak_blocks_used'] for r in results['continuous']),
        'max_waste_tokens_per_seq': max(r['max_waste_tokens_per_seq']
                                        for r in results['continuous']),
    }

    summary = results['summary']
    print("=== config: d_model={d_model} vocab={vocab_size} seqs={num_requests} "
          "prompt={prompt_tokens} new={max_new_tokens} blocks={num_blocks}x{block_size} "
          "trials={trials} ===".format(**results['config']))
    print("continuous throughput  median {:.0f} tok/s   range [{:.0f}, {:.0f}]".format(
        summary['continuous_tokens_per_sec_median'],
        *summary['continuous_tokens_per_sec_range']))
    print("static throughput      median {:.0f} tok/s   -> ratio {:.2f}x".format(
        summary['static_tokens_per_sec_median'],
        summary['throughput_ratio_continuous_vs_static']))
    print("TTFT first admitted    {:.1f} ms".format(summary['ttft_first_ms_median']))
    print("TTFT p50 / p90         {:.1f} / {:.1f} ms".format(
        summary['ttft_p50_ms_median'], summary['ttft_p90_ms_median']))
    print("mean inter-token lat.  {:.3f} ms".format(summary['mean_itl_ms_median']))
    print("per-token decode (1 seq)  paged {:.3f} ms   contiguous {:.3f} ms".format(
        summary['single_seq_paged_median_ms'], summary['single_seq_contiguous_median_ms']))
    print("static: first output after {:.1f} ms  ({:.1f}x later than continuous TTFT p90)".format(
        summary['static_total_time_ms_median'],
        summary['first_output_ratio_static_vs_continuous_p90']))
    print("capacity: paged {paged_concurrent_seqs} seqs vs contiguous "
          "{contiguous_concurrent_seqs} seqs  ({ratio:.2f}x) at {tokens_per_seq} tok/seq, "
          "{total_kv_slots} KV slots".format(**results['capacity']))
    print("peak blocks used {} / {}   max internal waste {:.2f} tokens/seq".format(
        summary['peak_blocks_used'], NUM_BLOCKS, summary['max_waste_tokens_per_seq']))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bench_results.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2, default=float)
    print("raw results written to", out_path)


if __name__ == "__main__":
    main()
