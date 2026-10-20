#!/usr/bin/env bash
# Create (or recreate with --force) the isolated conda env "gaden-bench".
#   bash bench/setup_env.sh [--force]
# Packages in ~/.local would otherwise shadow the env's own packages: pip skips
# them as "already satisfied" during creation, and python loads them at run time.
set -euo pipefail
cd "$(dirname "$0")"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
NAME=gaden-bench
if "$CONDA" env list | grep -qE "^$NAME\s"; then
  if [[ "${1:-}" == "--force" ]]; then "$CONDA" env remove -y -n "$NAME"; else echo "env $NAME exists (use --force)"; exit 1; fi
fi
"$CONDA" env create -f environment-gaden.yml
PREFIX="$("$CONDA" run -n "$NAME" python -c 'import sys; print(sys.prefix)')"
PYTHONNOUSERSITE=1 "$PREFIX/bin/python" -m pip install --no-user -r requirements-gaden.txt
# Keep ~/.local off sys.path even when the env's python is called without activation.
SITE="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "$SITE/sitecustomize.py" <<'PY'
# gaden-bench: never import from the user site (~/.local); see bench/setup_env.sh
import site, sys
_user = site.getusersitepackages()
sys.path[:] = [p for p in sys.path if not p.startswith(_user)]
PY
"$PREFIX/bin/python" - <<'PY'
import importlib, sys
bad = []
for m in ("numpy", "scipy", "matplotlib", "gymnasium", "warp", "pyvista", "pytest"):
    mod = importlib.import_module(m)
    ok = mod.__file__.startswith(sys.prefix)
    print(f"{m:11s} {getattr(mod, '__version__', '?'):8s} {'ok' if ok else 'NOT IN ENV: ' + mod.__file__}")
    bad += [] if ok else [m]
sys.exit(1 if bad else 0)
PY
echo "gaden-bench ready: $PREFIX"
