#!/bin/bash
# One-time setup for running Momentum Desk as an always-on systemd service
# on a Raspberry Pi. Run this from the repo root after cloning:
#
#   git clone <your-repo-url>
#   cd paper-trading
#   bash deploy/raspberrypi/setup_pi.sh
#
# This does NOT need to be re-run after every `git pull` -- only after a
# fresh clone, or if requirements.txt changed. To pick up a code update on
# an already-set-up Pi, just:
#
#   git pull
#   sudo systemctl restart papertrader
set -euo pipefail

if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    echo "Run this from the paper-trading repo root, e.g.:"
    echo "  bash deploy/raspberrypi/setup_pi.sh"
    exit 1
fi

REPO_DIR="$(pwd)"
SERVICE_USER="$(whoami)"

echo "== Momentum Desk: Raspberry Pi setup =="
echo "Repo: $REPO_DIR"
echo "Running as: $SERVICE_USER"
echo ""

# ---- 1. Swap file -------------------------------------------------------
# A Pi's usable RAM is often tighter than advertised once Flask/pandas/
# numpy are all loaded at once, so a swap file is cheap insurance rather
# than something to skip. Only add one if there isn't already enough.
CURRENT_SWAP_MB=$(free -m | awk '/^Swap:/{print $2}')
if [ "$CURRENT_SWAP_MB" -lt 1024 ]; then
    echo "Current swap: ${CURRENT_SWAP_MB}MB -- adding a 1GB swap file at /swapfile"
    if [ ! -f /swapfile ]; then
        sudo fallocate -l 1G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=1024
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
    fi
    sudo swapon /swapfile 2>/dev/null || true
    if ! grep -q '^/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    fi
else
    echo "Existing swap (${CURRENT_SWAP_MB}MB) already sufficient -- skipping."
fi
echo ""

# ---- 2. System packages --------------------------------------------------
echo "Installing python3-venv/pip (apt)..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip
echo ""

# ---- 3. Virtual environment + dependencies -------------------------------
# Raspberry Pi OS's pip is preconfigured to use piwheels.org as an extra
# index, so pandas/numpy install from precompiled ARM wheels instead of
# a slow (and on 512MB-1GB boards, often memory-starved) source build.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
echo "Installing Python dependencies (this can take a few minutes on a Pi)..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt
echo ""

# ---- 4. First-time admin account ----------------------------------------
if [ ! -f "data/auth_secrets.json" ]; then
    echo "First-time setup: create the dashboard's admin username and password."
    echo "(Two-factor authentication is set up separately, in the browser, the"
    echo "first time you log in.)"
    .venv/bin/python main.py setup-auth
fi
echo ""

# ---- 5. Install and start the systemd service ----------------------------
echo "Installing the systemd service..."
sudo cp deploy/raspberrypi/papertrader.service /etc/systemd/system/papertrader.service
sudo sed -i "s|__REPO_DIR__|$REPO_DIR|g; s|__USER__|$SERVICE_USER|g" /etc/systemd/system/papertrader.service
sudo systemctl daemon-reload
sudo systemctl enable papertrader
sudo systemctl restart papertrader

echo ""
echo "== Done =="
echo "Check status:  sudo systemctl status papertrader"
echo "View logs:     journalctl -u papertrader -f"
echo "Restart:       sudo systemctl restart papertrader"
echo ""
PI_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$PI_IP" ]; then
    echo "Once it's up, open http://$PI_IP:8000 from another device on the same network."
else
    echo "Once it's up, open http://<this-pi's-IP>:8000 from another device on the same network."
fi
