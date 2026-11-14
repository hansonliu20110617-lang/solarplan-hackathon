"""
One-shot query against a single external LLM provider's chat API, stdlib only
(urllib) so this never needs extra installs mid-hackathon. Keys come from
environment variables only -- never from argv or this file -- see
.env.example for which variable name each provider reads.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Only retry errors that are actually transient (overload/rate-limit/timeout)
# -- a bad key or bad model name is never going to succeed on retry, so those
# fail immediately instead of burning 3x the time on a call that can't work.
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1, 3)


def _load_dotenv():
    # Minimal .env loader (no python-dotenv dependency) -- only fills in
    # vars that aren't already set, so a real shell env var always wins.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# fast_model = cheap/quick tier (high-volume checks, boilerplate, routine
# review passes). quality_model = top tier (hard bugs, architecture calls,
# final pre-submission review). Pick with --quality; --model overrides both.
# Verified live against this project's own keys on 2026-11-05. OpenAI's
# GPT-6 family (Astra/Sol/Luna/Terra) superseded GPT-5 -- Luna is the
# fast/cheap tier, Astra the flagship; gpt-6.1-astra/luna and bare gpt-6-terra
# are not available on this key, so don't swap to those without re-checking.
# Gemini uses Google's "-latest" aliases (confirmed via the key's own
# ListModels call) so they auto-track Google's current flash/pro model
# instead of a pinned preview string going stale. Provider model IDs change
# often -- re-verify if these start failing.
PROVIDERS = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "fast_model": "gpt-6-luna",
        "quality_model": "gpt-6-astra",
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        "fast_model": "gemini-flash-latest",
        "quality_model": "gemini-pro-latest",
    },
    "xai": {
        "env": "XAI_API_KEY",
        "url": "https://api.x.ai/v1/chat/completions",
        "fast_model": "grok-4-fast",
        "quality_model": "grok-4",
    },
}


def _post(url, headers, payload, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    last_err = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_HTTP_CODES or attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}")
            last_err = f"HTTP {e.code}"
        except TimeoutError:
            if attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError("timed out after retries")
            last_err = "timeout"
        wait = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
        print(f"  ({last_err}, retrying in {wait}s -- attempt {attempt + 2}/{_MAX_ATTEMPTS})", flush=True)
        time.sleep(wait)


# Verified live on 2026-11-05: for a trivial prompt, Gemini's hidden
# "thinking" tokens are billed as output (92 thought tokens vs 0 with the
# budget capped -- a 16x difference on nothing). OpenAI's reasoning_tokens
# dropped from 6 to 0 going from unset to reasoning_effort="none". xAI only
# responded to "none" specifically -- "low"/"high"/unset all cost about the
# same ~105-122 reasoning tokens, so "low" is not actually cheaper there.
#
# Verified live on 2026-11-14: reasoning-token usage at quality tier isn't
# deterministic -- the identical prompt/settings sometimes used 512 reasoning
# tokens (fine) and sometimes used the full budget, leaving 0 tokens for the
# actual answer. 1200 cut it too close for a real code-review prompt; 3000
# still wasn't enough for Gemini on a complex multi-function file review
# (2878 thinking + 118 output = 2996, truncated mid-sentence) -- bumped to
# 5000. See the truncation checks in ask() below for what happens if it
# still happens (fail loudly, don't silently return a cut-off answer).
FAST_MAX_TOKENS = 400
QUALITY_MAX_TOKENS = 5000


def _print_usage(provider, model, usage_line):
    print(f"  [usage] {provider}/{model}: {usage_line}", flush=True)


def ask(provider, prompt, model=None, quality=False):
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["env"])
    if not key:
        raise RuntimeError(f"{cfg['env']} is not set -- see .env.example")
    model = model or (cfg["quality_model"] if quality else cfg["fast_model"])
    max_tokens = QUALITY_MAX_TOKENS if quality else FAST_MAX_TOKENS

    if provider == "gemini":
        url = cfg["url"].format(model=model, key=key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingBudget": -1 if quality else 0}},
        }
        data = _post(url, {"Content-Type": "application/json"}, payload)
        um = data.get("usageMetadata", {})
        _print_usage(provider, model,
                     f"{um.get('promptTokenCount', '?')} prompt + "
                     f"{um.get('candidatesTokenCount', '?')} output "
                     f"({um.get('thoughtsTokenCount', 0)} thinking) = "
                     f"{um.get('totalTokenCount', '?')} total")
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            # Same failure mode as the openai/xai branch below: thinking
            # tokens can eat the whole budget and leave a truncated (or
            # empty) answer that silently looks like a short-but-complete
            # response instead of a cut-off one.
            raise RuntimeError(
                f"{provider}/{model}: hit MAX_TOKENS ({max_tokens} budget) -- response "
                "was truncated mid-answer, not a real complete-but-short result"
            )
        return candidate["content"]["parts"][0]["text"]

    # openai and xai both speak the same chat-completions request/response shape,
    # except openai's newer models reject max_tokens in favor of max_completion_tokens
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "high" if quality else "none",
    }
    payload["max_completion_tokens" if provider == "openai" else "max_tokens"] = max_tokens
    data = _post(cfg["url"], headers, payload)
    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    details = usage.get("completion_tokens_details", {}) or {}
    _print_usage(provider, model,
                 f"{usage.get('prompt_tokens', '?')} prompt + "
                 f"{usage.get('completion_tokens', '?')} completion "
                 f"({details.get('reasoning_tokens', 0)} reasoning) = "
                 f"{usage.get('total_tokens', '?')} total")
    if not content.strip() and choice.get("finish_reason") == "length":
        # Reasoning-token usage isn't deterministic -- the same prompt can
        # occasionally consume the whole budget on invisible reasoning and
        # leave nothing for the actual answer. Silently returning "" would
        # look exactly like "model found nothing to report" to a caller
        # (see peer_review.py) -- fail loudly instead so it's retried with
        # more headroom rather than mistaken for a clean review.
        raise RuntimeError(
            f"{provider}/{model}: reasoning consumed the entire {max_tokens}-token "
            "budget, leaving no content -- retry with --model or a larger cap, "
            "not a real 'no issues found' result"
        )
    return content


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", choices=PROVIDERS)
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None, help="overrides fast_model/quality_model entirely")
    ap.add_argument("--quality", action="store_true", help="use the top-tier model instead of the fast/cheap one")
    args = ap.parse_args()
    try:
        print(ask(args.provider, args.prompt, args.model, args.quality))
    except RuntimeError as e:
        print(f"error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
