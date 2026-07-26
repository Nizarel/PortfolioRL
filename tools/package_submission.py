r"""Build the submission archive.

Run:  .\.venv\Scripts\python.exe tools\package_submission.py

The archive is assembled from an explicit allow-list rather than by excluding
things, because the failure mode that matters is shipping a zip that is missing
a figure the report references -- not one that is slightly too large. Every file
the report or the notebooks depend on is either included or reported as missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (glob, required) -- required patterns that match nothing are reported as errors
INCLUDE: tuple[tuple[str, bool], ...] = (
    ("README.md", True),
    ("pyproject.toml", True),
    ("Docs/*.md", True),
    ("Docs/*.png", False),
    ("notebooks/*.ipynb", True),
    ("src/portfoliorl/*.py", True),
    ("tests/*.py", True),
    ("tools/*.py", False),
    ("tools/*.ps1", False),
    ("artifacts/figures/*.png", True),
    ("artifacts/results/*.csv", True),
    ("artifacts/results/*.json", True),
    ("artifacts/models/*.pt", True),
    ("data/processed/*.csv", True),
)

# Artefacts the written report and the run-all notebook rely on by name. If one
# of these is absent the package is incomplete even though the zip would build.
REQUIRED_FILES: tuple[str, ...] = (
    "artifacts/results/02_benchmark_scorecard_test.csv",
    "artifacts/results/03_training_run.json",
    "artifacts/results/03_validation_scorecard.csv",
    "artifacts/results/04_best_config.json",
    "artifacts/results/04_coarse_grid.csv",
    "artifacts/results/05_test_scorecard.csv",
    "artifacts/models/05_headline_seed0.pt",
)

EXPECTED_NOTEBOOKS: tuple[str, ...] = (
    "notebooks/00_run_all.ipynb",
    "notebooks/01_data_eda.ipynb",
    "notebooks/02_env_benchmarks.ipynb",
    "notebooks/03_dqn_training.ipynb",
    "notebooks/04_tuning.ipynb",
    "notebooks/05_results_ablation.ipynb",
)


def collect() -> tuple[list[Path], list[str]]:
    """Resolve the allow-list into concrete files, reporting empty patterns."""
    files: list[Path] = []
    problems: list[str] = []

    for pattern, required in INCLUDE:
        matched = sorted(p for p in ROOT.glob(pattern) if p.is_file())
        if required and not matched:
            problems.append(f"pattern matched nothing: {pattern}")
        files.extend(matched)

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).exists():
            problems.append(f"missing required artefact: {relative}")

    # de-duplicate while preserving order
    seen: set[Path] = set()
    unique = [p for p in files if not (p in seen or seen.add(p))]
    return unique, problems


def check_notebooks_have_outputs() -> list[str]:
    """A notebook shipped without outputs asks the grader to run it themselves."""
    problems: list[str] = []
    for relative in EXPECTED_NOTEBOOKS:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"missing notebook: {relative}")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in blob["cells"] if c["cell_type"] == "code"]
        without = [c for c in code_cells if not c.get("outputs")]
        if not code_cells:
            problems.append(f"{relative} contains no code cells")
        elif len(without) > len(code_cells) * 0.2:
            problems.append(
                f"{relative}: {len(without)} of {len(code_cells)} code cells have no "
                "output -- was it executed?"
            )
    return problems


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, files: list[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())

        manifest = [
            {
                "path": p.relative_to(ROOT).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in files
        ]
        archive.writestr(
            "MANIFEST.json",
            json.dumps(
                {
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "file_count": len(manifest),
                    "total_bytes": sum(m["bytes"] for m in manifest),
                    "files": manifest,
                },
                indent=2,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "PortfolioRL_submission.zip",
        help="path of the archive to write",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="build the archive even if required artefacts are missing",
    )
    args = parser.parse_args()

    files, problems = collect()
    problems.extend(check_notebooks_have_outputs())

    if problems:
        print("problems found:")
        for problem in problems:
            print(f"  - {problem}")
        if not args.allow_incomplete:
            print("\nrefusing to build an incomplete archive; "
                  "pass --allow-incomplete to override")
            return 1
        print()

    build(args.output, files)

    size_mb = args.output.stat().st_size / 1e6
    by_kind: dict[str, int] = {}
    for path in files:
        by_kind[path.suffix or "(none)"] = by_kind.get(path.suffix or "(none)", 0) + 1

    print(f"wrote {args.output.relative_to(ROOT)}")
    print(f"  {len(files)} files, {size_mb:,.1f} MB compressed")
    for suffix, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
