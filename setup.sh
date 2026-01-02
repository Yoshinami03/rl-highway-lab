#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for this project:
# - ensure pyenv Python is available
# - pin local Python to the desired version
# - create .venv with that Python
# - install dependencies

PY_VER="3.13.7"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found. Install pyenv first, then rerun this script." >&2
  exit 1
fi

echo "==> Ensuring Python ${PY_VER} is installed via pyenv..."
pyenv install -s "${PY_VER}"

echo "==> Setting local pyenv version..."
(
  cd "${REPO_ROOT}"
  pyenv local "${PY_VER}"
)

echo "==> Creating virtualenv with Python ${PY_VER}..."
(
  cd "${REPO_ROOT}"
  PYENV_VERSION="${PY_VER}" python -m venv .venv
)

echo "==> Upgrading pip/wheel and installing requirements..."
"${REPO_ROOT}/.venv/bin/pip" install --upgrade pip wheel
"${REPO_ROOT}/.venv/bin/pip" install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "Setup complete."
echo "To use: source .venv/bin/activate"
