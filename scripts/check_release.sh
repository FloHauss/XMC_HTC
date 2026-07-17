#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"

python -m unittest discover -s tests -v
python -m compileall -q \
  integrations/hgclr \
  XMLmodels/CascadeXML \
  XMLPreprocessing \
  htc/hbgl \
  dataset_transfer \
  tests

find integrations/hgclr -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0 \
  | xargs -0 -r -n1 bash -n
bash -n XMLmodels/pecos/job_*.sh
bash -n XMLmodels/pecos/run_ensemble/*.sh
bash -n htc/hbgl/scripts/*.sh

python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path("XMLmodels/pecos/run_ensemble/params").rglob("*.json"))
if not paths:
    raise SystemExit("No XR-Transformer parameter files found")
for path in paths:
    with path.open(encoding="utf-8") as stream:
        json.load(stream)
print(f"Parsed {len(paths)} XR-Transformer parameter files")
PY

echo "Bounded release checks passed."
