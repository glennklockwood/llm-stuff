"""
Speculative Decoding — Worked Example

Both models must share a tokenizer and vocabulary, which is required for the
acceptance math to work: the probability of a draft token must be comparable
between the two models' distributions.

Usage:
    python speculative_decode.py [--draft-model gpt2] [--full-model gpt2-large]
                                 [--sample] [--K 4] [--max-new-tokens 40] [--prompt "..."]

    --draft-model  HuggingFace model name for the draft (cheap) model (default: gpt2)
    --full-model   HuggingFace model name for the full (expensive) model (default: gpt2-large)
    --sample       Switch from greedy (argmax) to multinomial sampling.
                   Greedy mode is deterministic and produces token-for-token
                   identical output between baseline and speculative runs, which
                   proves correctness.  Sampling mode uses the proper rejection
                   criterion (accept with prob min(1, p_full/p_draft); correct
                   from relu(p_full - p_draft)) so the output distribution
                   matches the full model's but individual sequences will differ.
"""

import argparse
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def run_baseline(model, input_ids, max_new_tokens, device, greedy=True):
    """
    Standard autoregressive decode — one token per full-model call.

    greedy=True  : argmax (deterministic, identical output every run)
    greedy=False : multinomial sampling from the full model's distribution

    Returns:
        tokens        : list of generated token ids (length == max_new_tokens)
        n_full_passes : number of model forward passes (prefill + decode)
        elapsed_s     : wall time in seconds
    """
    sync(device)
    t0 = time.perf_counter()

    generated = []
    logits, kv = forward(model, input_ids)
    n_passes = 1   # prefill

    while len(generated) < max_new_tokens:
        probs = F.softmax(logits[:, -1, :], dim=-1)   # [1, vocab_size]
        if greedy:
            token = probs.argmax(dim=-1, keepdim=True)            # [1, 1]
        else:
            token = torch.multinomial(probs, num_samples=1)       # [1, 1]
        generated.append(token.item())
        if len(generated) >= max_new_tokens:
            break
        logits, kv = forward(model, token, kv)
        n_passes += 1

    sync(device)
    elapsed = time.perf_counter() - t0

    return generated, n_passes, elapsed


def run_speculative(draft_model, full_model, input_ids, max_new_tokens, device, K=4, greedy=True):
    """
    Speculative decode with a choice of acceptance strategy.

    greedy=True  : accept iff draft argmax == full model argmax; output is
                   token-for-token identical to run_baseline(..., greedy=True).
    greedy=False : proper rejection sampling — accept draft token t with
                   probability min(1, p_full[t] / p_draft[t]); on rejection
                   draw a correction from relu(p_full - p_draft).  Output
                   distribution is identical to run_baseline(..., greedy=False)
                   but individual sequences will differ.

    Returns:
        tokens         : list of generated token ids (length == max_new_tokens)
        n_full_passes  : forward passes on the (expensive) full model
        n_draft_passes : forward passes on the (cheap) draft model
        elapsed_s      : wall time in seconds
        rounds         : per-round stats for inspection
    """
    sync(device)
    t0 = time.perf_counter()

    generated    = []
    context_len  = input_ids.shape[1]
    rounds       = []

    draft_logits, draft_kv = forward(draft_model, input_ids)
    full_logits,  full_kv  = forward(full_model,  input_ids)
    n_full_passes  = 1
    n_draft_passes = 1

    while len(generated) < max_new_tokens:

        # 1. Draft K tokens
        draft_tokens = []
        draft_probs  = []   # needed for rejection sampling acceptance criterion
        cur_logits   = draft_logits[:, -1, :]
        # cur_logits is [1, vocab_size] — a single row sliced from draft_logits
        # which is [1, seq_len, vocab_size]. The [:, -1, :] takes the last
        # position across the batch and all vocab dimensions, dropping the
        # sequence dimension.

        for _ in range(K):
            probs = F.softmax(cur_logits, dim=-1)              # [1, vocab_size]
            if greedy:
                token = probs.argmax(dim=-1, keepdim=True)     # [1, 1]
            else:
                token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(token.item())
            draft_probs.append(probs[0])                       # [vocab_size]
            cur_logits, draft_kv = forward(draft_model, token, draft_kv)
            cur_logits = cur_logits[:, -1, :]   # [1, 1, vocab_size] -> [1, vocab_size]
            n_draft_passes += 1

        # 2. Full model verifies all K draft tokens in one batched pass
        draft_tensor = torch.tensor([[*draft_tokens]], device=input_ids.device)
        verify_logits, full_kv_verify = forward(full_model, draft_tensor, full_kv)
        n_full_passes += 1

        # Calculate full model's distribution at each draft position
        # First element: softmax over last token of full_logits
        full_probs = []
        first_prob = F.softmax(full_logits[:, -1, :], dim=-1)[0]
        full_probs.append(first_prob)

        # Remaining elements: softmax over tokens 0..K-2 of verify_logits
        for i in range(K - 1):
            prob = F.softmax(verify_logits[:, i, :], dim=-1)[0]
            full_probs.append(prob)
        bonus_probs = F.softmax(verify_logits[:, -1, :], dim=-1)[0]   # [vocab_size]

        # 3. Accept or reject each draft token
        accepted         = []
        n_draft_accepted = 0
        all_accepted     = True

        for draft_tok, p_draft, p_full in zip(draft_tokens, draft_probs, full_probs):
            if greedy:
                if draft_tok == p_full.argmax().item():
                    accepted.append(draft_tok)
                    n_draft_accepted += 1
                else:
                    accepted.append(p_full.argmax().item())   # corrected token
                    all_accepted = False
                    break
            else:
                accept_prob = min(1.0, (p_full[draft_tok] / p_draft[draft_tok]).item())
                if torch.rand(1).item() < accept_prob:
                    accepted.append(draft_tok)
                    n_draft_accepted += 1
                else:
                    # correction: sample from the residual distribution
                    adjusted  = F.relu(p_full - p_draft)
                    corrected = torch.multinomial(adjusted / adjusted.sum(), 1).item()
                    accepted.append(corrected)
                    all_accepted = False
                    break

        if all_accepted:
            if greedy:
                accepted.append(bonus_probs.argmax().item())
            else:
                accepted.append(torch.multinomial(bonus_probs.unsqueeze(0), 1).item())

        generated.extend(accepted)
        rounds.append({
            "drafted":        K,
            "draft_accepted": n_draft_accepted,
            "total_accepted": len(accepted),
            "got_bonus":      all_accepted,
        })

        context_len += len(accepted)

        if len(generated) >= max_new_tokens:
            break

        # 4. Sync KV caches for next round
        last_token = torch.tensor([[accepted[-1]]], device=input_ids.device)

        # rolls back both caches to only cover the positions that were genuinely
        # accepted, discarding the rejected tail
        full_kv      = truncate_kv(full_kv_verify, context_len - len(accepted) + n_draft_accepted)
        full_logits,  full_kv  = forward(full_model,  last_token, full_kv)
        n_full_passes += 1

        # repeat for the draft model
        draft_kv     = truncate_kv(draft_kv, context_len - len(accepted) + n_draft_accepted)
        draft_logits, draft_kv = forward(draft_model, last_token, draft_kv)
        n_draft_passes += 1

    sync(device)
    elapsed = time.perf_counter() - t0

    return generated[:max_new_tokens], n_full_passes, n_draft_passes, elapsed, rounds


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument("--draft-model",     type=str,   default="gpt2",
                        help="HuggingFace model name for the draft model (default: gpt2)")
    parser.add_argument("--full-model",      type=str,   default="gpt2-xl",
                        help="HuggingFace model name for the full model (default: gpt2-xl)")
    parser.add_argument("--sample",          action="store_true",
                        help="Use multinomial sampling instead of greedy (argmax) decoding. "
                             "Sampling produces non-deterministic output so the two runs will "
                             "not produce identical sequences.")
    parser.add_argument("--K",               type=int,   default=4,
                        help="Number of draft tokens to propose per round (default: 4)")
    parser.add_argument("--max-new-tokens",  type=int,   default=40,
                        help="Number of new tokens to generate (default: 40)")
    parser.add_argument("--prompt",          type=str,   default="The quick brown fox",
                        help="Prompt string (default: 'The quick brown fox')")
    args = parser.parse_args()

    greedy           = not args.sample
    K                = args.K
    MAX_NEW_TOKENS   = args.max_new_tokens
    prompt           = args.prompt
    draft_model_name = args.draft_model
    full_model_name  = args.full_model
    mode_label       = "greedy" if greedy else "sampling"

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    tokenizer   = GPT2Tokenizer.from_pretrained(draft_model_name)
    dev = torch.device(device)
    draft_model = GPT2LMHeadModel.from_pretrained(draft_model_name).to(dev).eval()  # type: ignore[arg-type]
    full_model  = GPT2LMHeadModel.from_pretrained(full_model_name).to(dev).eval()   # type: ignore[arg-type]

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    N         = input_ids.shape[1]
    print(f"Prompt ({N} tokens): {prompt!r}")
    print(f"Generating {MAX_NEW_TOKENS} tokens, K={K} draft tokens per round, mode={mode_label}.\n")

    base_tokens, base_passes, base_time = run_baseline(
        full_model, input_ids, MAX_NEW_TOKENS, device, greedy=greedy
    )
    base_text = tokenizer.decode(base_tokens)

    print(f"Baseline (full model only, {mode_label})")
    print(f"  output              : {base_text!r}")
    print(f"  full-model passes   : {base_passes}")
    print(f"  tokens/pass         : {MAX_NEW_TOKENS / base_passes:.2f}")
    print(f"  wall time           : {base_time:.3f}s")
    print(f"  tokens/s            : {MAX_NEW_TOKENS / base_time:.1f}")

    print()

    spec_tokens, spec_full_passes, spec_draft_passes, spec_time, rounds = run_speculative(
        draft_model, full_model, input_ids, MAX_NEW_TOKENS, device=device, K=K, greedy=greedy
    )
    spec_text = tokenizer.decode(spec_tokens)

    total_draft_accepted = sum(r["draft_accepted"] for r in rounds)
    total_tokens_gen     = sum(r["total_accepted"] for r in rounds)
    bonus_rounds         = sum(1 for r in rounds if r["got_bonus"])

    print(f"Speculative decode (draft={draft_model_name}, full={full_model_name}, {mode_label})")
    print(f"  output              : {spec_text!r}")
    n_extend = spec_full_passes - 1 - len(rounds)   # prefill=1, verify=rounds, rest=extend
    print(f"  full-model passes   : {spec_full_passes}  ({len(rounds)} verify + {n_extend} extend + 1 prefill)")
    print(f"  tokens/pass         : {MAX_NEW_TOKENS / spec_full_passes:.2f}  (verify batches {K} tokens at once)")
    print(f"  draft-model passes  : {spec_draft_passes}")
    print(f"  wall time           : {spec_time:.3f}s")
    print(f"  tokens/s            : {MAX_NEW_TOKENS / spec_time:.1f}")
    print(f"  rounds              : {len(rounds)}")
    print(f"  bonus tokens        : {bonus_rounds}/{len(rounds)} rounds")
    print(f"  draft acceptance    : {total_draft_accepted}/{total_tokens_gen} "
          f"({total_draft_accepted/total_tokens_gen:.0%})")

    print()

    speedup        = base_time / spec_time
    pass_reduction = 1 - spec_full_passes / base_passes

    print("Comparison")
    if greedy:
        outputs_match = (base_tokens == spec_tokens)
        print(f"  outputs identical        : {outputs_match}")
        if not outputs_match:
            for i, (a, b) in enumerate(zip(base_tokens, spec_tokens)):
                if a != b:
                    print(f"  first divergence at token {i}: "
                          f"baseline={tokenizer.decode([a])!r}  "
                          f"speculative={tokenizer.decode([b])!r}")
                    break
    else:
        print("  outputs identical        : n/a (sampling is non-deterministic)")
    print(f"  wall-time speedup        : {speedup:.2f}x  ({base_time:.3f}s vs {spec_time:.3f}s)")
    print(f"  full-model pass reduction: {pass_reduction:.0%}  ({base_passes} vs {spec_full_passes} passes)")


def forward(model, input_ids, past_kv=None):
    """
    One forward pass.

    input_ids : [1, seq_len]  — one or more tokens
    past_kv   : KV cache covering all prior positions (or None for full prefill)

    Returns:
        logits   : [1, seq_len, vocab_size]
                   logits[:, i, :] = distribution for the token *after* input_ids[i],
                   given all prior context
        past_kv  : updated KV cache, now covering prior + seq_len positions
    """
    with torch.no_grad():
        out = model(input_ids, past_key_values=past_kv, use_cache=True)
    return out.logits, out.past_key_values


def truncate_kv(past_kv, seq_len):
    """
    Trim the KV cache to seq_len positions, in-place.

    Handles three formats across transformers versions:
      - Legacy tuple-of-tuples: ((k0, v0), (k1, v1), ...)
      - DynamicCache with crop() method (transformers >= ~4.44)
      - DynamicCache with key_cache/value_cache lists (transformers ~4.36-4.43)
    """
    if isinstance(past_kv, tuple):
        return tuple(
            (k[..., :seq_len, :], v[..., :seq_len, :])
            for k, v in past_kv
        )
    if hasattr(past_kv, 'crop'):
        past_kv.crop(seq_len)
        return past_kv
    for i in range(len(past_kv.key_cache)):
        past_kv.key_cache[i]   = past_kv.key_cache[i][..., :seq_len, :]
        past_kv.value_cache[i] = past_kv.value_cache[i][..., :seq_len, :]
    if hasattr(past_kv, '_seen_tokens'):
        past_kv._seen_tokens = seq_len
    return past_kv


def sync(device):
    """Block until all pending device ops are complete (for accurate timing)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

if __name__ == "__main__":
    main()