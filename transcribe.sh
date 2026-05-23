export PATH="$(dirname "$0")/.local/bin:$PATH"
export NLTK_DATA="$(dirname "$0")/.local/nltk_data"
.local/venv/bin/python transcribe.py "$1"
