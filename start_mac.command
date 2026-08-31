#!/bin/bash
# One-click launcher for macOS: double-click this file in Finder (or run
# `./start_mac.command` from Terminal). First run sets up a virtual
# environment and installs dependencies; every run after that starts
# straight up. Requires Python 3 to already be installed on the Mac --
# this only automates the venv/pip/run steps, it does not bundle Python
# itself into a standalone .app.
cd "$(dirname "$0")" || exit 1

pause_and_exit() {
    echo ""
    read -r -p "Press Enter to close this window..."
    exit "${1:-0}"
}

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 wasn't found on this Mac."
    echo "Install it from https://www.python.org/downloads/ (or run 'brew install python3'"
    echo "if you use Homebrew), then double-click this file again."
    pause_and_exit 1
fi

if [ ! -d ".venv" ]; then
    echo "First run: creating a virtual environment and installing dependencies..."
    echo "(This only happens once -- every run after this starts straight up.)"
    python3 -m venv .venv || pause_and_exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt || pause_and_exit 1

if [ ! -f "data/auth_secrets.json" ]; then
    echo ""
    echo "First-time setup: create the dashboard's admin username and password."
    echo "(Two-factor authentication is set up separately, in the browser, the"
    echo "first time you log in.)"
    python main.py setup-auth || pause_and_exit 1
fi

# Open the dashboard in the default browser a couple of seconds after the
# server starts, instead of racing it (a request against a port nothing is
# listening on yet would just fail).
( sleep 2 && open "http://127.0.0.1:8000" ) &

echo ""
echo "Starting Momentum Desk -- http://127.0.0.1:8000"
echo "Leave this window open while it's trading. Press Ctrl+C here to stop it."
echo ""
python main.py serve

pause_and_exit 0
