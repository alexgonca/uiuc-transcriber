export PATH="$(dirname "$0")/.local/bin:$PATH"
.local/venv/bin/python transcribe.py "$1"
