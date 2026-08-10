"""CLI demo for the multi-agent coding team."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.orchestrator import run_team


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    instruction = " ".join(argv).strip() or "Write a function that deduplicates a list while preserving order"

    state = run_team(instruction)
    print("=== PLAN ===")
    for i, step in enumerate(state.plan, 1):
        print(f"{i}. {step}")

    print("\n=== CODE ===")
    print(state.code)

    print("\n=== CRITIC ===")
    if state.critiques:
        print("REJECTED after revisions=", state.revision)
        for c in state.critiques:
            print("-", c)
        return 1

    print("APPROVED after revisions=", state.revision)
    out = ROOT / "demo" / "last_output.py"
    out.write_text(state.code, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
