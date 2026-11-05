"""Bake pipeline/validation_results.json into interface/template.html via a
simple string-substitution placeholder -- no backend, works from file://.

Assumptions:
- template.html contains the literal placeholder token __RESULTS_JSON__
  inside `window.RESULTS_DATA = __RESULTS_JSON__;`.
- validation_results.json already exists (produced by validate_template.py);
  this script never recomputes or fabricates results, so it fails loudly
  instead of silently exporting garbage when that file is missing.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RESULTS_PATH = os.path.join(HERE, "validation_results.json")
METRICS_PATH = os.path.join(HERE, "metrics_summary.json")
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "interface", "template.html")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "interface", "result.html")

PLACEHOLDER = "__RESULTS_JSON__"


def build_payload():
    if not os.path.exists(RESULTS_PATH):
        raise SystemExit(
            f"Missing {RESULTS_PATH}. Run validate_template.py first to "
            "produce validation_results.json before exporting."
        )

    with open(RESULTS_PATH) as f:
        payload = json.load(f)

    # metrics_summary.json is optional context from the training stage --
    # include it if present, but never fail export for lacking it.
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            payload["metrics_summary"] = json.load(f)

    return payload


def export(payload):
    if not os.path.exists(TEMPLATE_PATH):
        raise SystemExit(f"Missing template at {TEMPLATE_PATH}.")

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()

    if PLACEHOLDER not in html:
        raise SystemExit(
            f"{TEMPLATE_PATH} has no {PLACEHOLDER} token to replace -- check the template."
        )

    html = html.replace(PLACEHOLDER, json.dumps(payload, indent=2))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    return OUTPUT_PATH


if __name__ == "__main__":
    data = build_payload()
    out_path = export(data)
    print(f"Saved {out_path}")
