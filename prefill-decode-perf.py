import requests

def run_inference(label, prompt):
    print(f"\n{'='*50}")
    print(f"Test: {label}")
    print(f"Prompt length: {len(prompt.split())} words")
    print("Running...")

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()
    prefill_toks = data['prompt_eval_count']
    prefill_time = data['prompt_eval_duration'] / 1e9
    decode_toks = data['eval_count']
    decode_time = data['eval_duration'] / 1e9

    print(f"Response snippet: {data['response'][:100]}...")
    print(f"  Prompt tokens : {prefill_toks:>6}   time: {prefill_time:>6.2f}s   speed: {prefill_toks/prefill_time:>7.1f} tok/s")
    print(f"  Output tokens : {decode_toks:>6}   time: {decode_time:>6.2f}s   speed: {decode_toks/decode_time:>7.1f} tok/s")

run_inference(
    label="Short prompt, long output (decode-bound)",
    prompt="Write a 500 word essay about photosynthesis."
)

run_inference(
    label="Long prompt, short output (prefill-bound)",
    prompt="repeat this word " * 500 + " How many words did I just send?"
)
