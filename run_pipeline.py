"""
Chains the five pipeline stages as subprocesses, not library imports -- each
stage is meant to stay a standalone, swappable script (simulate/preprocess
can be rewritten entirely for the real track without this driver changing).
Stops at the first stage that fails instead of limping forward on bad data.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"

STAGES = [
    ("simulate", PIPELINE / "simulate_template.py"),
    ("preprocess", PIPELINE / "preprocess_template.py"),
    ("train", PIPELINE / "train_template.py"),
    ("validate", PIPELINE / "validate_template.py"),
    ("export", PIPELINE / "export_template.py"),
]


def run_stage(name, script_path):
    print(f"\n=== {name} ({script_path.name}) ===", flush=True)
    result = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\nStage '{name}' failed (exit {result.returncode}) -- stopping.", flush=True)
        sys.exit(result.returncode)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, script_path in STAGES:
        if only and name != only:
            continue
        if not script_path.exists():
            print(f"Skipping '{name}': {script_path} not written yet.", flush=True)
            continue
        run_stage(name, script_path)
    result_html = ROOT / "interface" / "result.html"
    if result_html.exists():
        print(f"\nDone. Open {result_html} to view results.", flush=True)
    else:
        print("\nDone (export stage didn't produce interface/result.html -- check its output above).", flush=True)


if __name__ == "__main__":
    main()
