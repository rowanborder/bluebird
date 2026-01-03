#!/bin/bash
# Install and enable bluebird-camera systemd service

set -e

SERVICE_FILE="bluebird-camera.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_FILE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Bluebird Camera service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root (use sudo)"
   exit 1
fi

if [ -f "$SERVICE_PATH" ]; then
   systemctl stop "$SERVICE_FILE"
   systemctl disable "$SERVICE_FILE"
fi

# Copy service file
echo "Copying service file to $SERVICE_PATH..."
cp "$SCRIPT_DIR/$SERVICE_FILE" "$SERVICE_PATH"
chmod 644 "$SERVICE_PATH"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service to start at boot
echo "Enabling service to start at boot..."
systemctl enable "$SERVICE_FILE"

# Start the service
echo "Starting service..."
systemctl start "$SERVICE_FILE"

echo ""
echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  Start:    sudo systemctl start bluebird-camera"
echo "  Stop:     sudo systemctl stop bluebird-camera"
echo "  Status:   sudo systemctl status bluebird-camera"
echo "  Logs:     sudo journalctl -u bluebird-camera -f"
echo "  Enable:   sudo systemctl enable bluebird-camera"
echo "  Disable:  sudo systemctl disable bluebird-camera"
echo ""
echo "View current status:"
systemctl status bluebird-camera --no-pager
