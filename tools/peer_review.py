"""
Sends one file's contents to one or more external-model providers concurrently
with a fixed "find real bugs, not style" prompt, and writes every provider's
raw response to peer_review_results.json. This script only collects opinions
-- it never auto-applies anything; reconciling disagreement is the
orchestrator's (a person's, or Claude's) job, not this script's.
"""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask_llm import PROVIDERS, ask  # noqa: E402

REVIEW_PROMPT = """You are reviewing code for a hackathon team. Find real bugs,
edge cases, or correctness problems only -- not style preferences. If you see
no real issues, say so plainly. Keep your answer under 200 words.

--- file: {name} ---
{code}
"""


def review_file(path, providers, model=None, quality=False):
    code = Path(path).read_text()
    prompt = REVIEW_PROMPT.format(name=Path(path).name, code=code)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(ask, p, prompt, model, quality): p for p in providers}
        for future, provider in futures.items():
            try:
                results[provider] = future.result()
            except Exception as e:
                results[provider] = f"ERROR: {e}"
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("providers", help="comma-separated, e.g. openai,xai")
    ap.add_argument("--quality", action="store_true", help="use each provider's top-tier model (final/pre-submission review) instead of the fast/cheap tier (routine passes)")
    args = ap.parse_args()

    providers = [p for p in args.providers.split(",") if p in PROVIDERS]
    if not providers:
        print(f"no valid providers in '{args.providers}' -- choices: {list(PROVIDERS)}")
        sys.exit(1)

    results = review_file(args.file, providers, quality=args.quality)
    out_path = Path("peer_review_results.json")
    out_path.write_text(json.dumps({"file": args.file, "quality_tier": args.quality, "reviews": results}, indent=2))
    for provider, text in results.items():
        print(f"\n=== {provider} ===\n{text}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
