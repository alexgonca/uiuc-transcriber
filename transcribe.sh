#!/bin/bash
# Auto-update so infrequent users always run the latest fixes. Everything is
# wrapped in a function so bash reads the whole script before executing: the
# pull below may rewrite this file mid-run, and without this bash would resume
# at a stale byte offset and corrupt. Set TRANSCRIBE_NO_UPDATE=1 to skip.
main() {
    local here
    here="$(dirname "$0")"

    if [ -z "$TRANSCRIBE_NO_UPDATE" ] && git -C "$here" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Checking for updates..."
        git -C "$here" pull --ff-only \
            || echo "WARNING: could not update (offline or local changes); continuing with current version."
    fi

    export PATH="$here/.local/bin:$PATH"
    export NLTK_DATA="$here/.local/nltk_data"
    .local/venv/bin/python transcribe.py "$1"
}
main "$@"
