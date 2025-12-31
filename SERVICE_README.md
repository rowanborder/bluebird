# Bluebird Camera Service

This directory contains systemd service files to run the bluebird camera classifier as a background service that starts automatically at boot.

## Installation

1. Copy the script to your Pi:
   ```bash
   scp bluebird_cam.py pi@<pi-ip>:/home/pi/
   scp bluebird-camera.service pi@<pi-ip>:/home/pi/
   scp install-service.sh pi@<pi-ip>:/home/pi/
   ```

2. SSH into your Pi and run the installer:
   ```bash
   ssh pi@<pi-ip>
   cd /home/pi
   sudo bash install-service.sh
   ```

## Configuration

Edit `/etc/systemd/system/bluebird-camera.service` to adjust:
- **Working directory**: Change `WorkingDirectory=/home/pi` if needed
- **Script arguments**: Modify the `ExecStart` line with your model paths
- **Restart policy**: Adjust `RestartSec` (delay between restarts) and `StartLimitBurst` (max restarts)
- **User**: Change `User=pi` if running under a different user

After editing, reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart bluebird-camera
```

## Manual Service Control

```bash
# Start the service
sudo systemctl start bluebird-camera

# Stop the service
sudo systemctl stop bluebird-camera

# Check status
sudo systemctl status bluebird-camera

# View live logs
sudo journalctl -u bluebird-camera -f

# View last 50 lines of logs
sudo journalctl -u bluebird-camera -n 50

# Disable auto-start (but allow manual start)
sudo systemctl disable bluebird-camera

# Re-enable auto-start
sudo systemctl enable bluebird-camera
```

## Service Features

- **Auto-restart**: Service automatically restarts on failure
- **Restart delay**: 10 seconds between restart attempts
- **Burst limit**: Allows up to 5 restarts within 600 seconds; stops trying if exceeded
- **Auto-boot**: Service starts automatically when Pi boots
- **Logging**: All output goes to systemd journal (viewable with `journalctl`)
- **Resource limits**: 
  - Memory: Limited to 1GB
  - CPU: Limited to 80%

## Troubleshooting

### Service won't start
Check the logs:
```bash
sudo journalctl -u bluebird-camera -n 50
```

### Service keeps restarting
Look at the restart count in the logs. If it exceeds `StartLimitBurst`, the service will stop trying. Reset with:
```bash
sudo systemctl reset-failed bluebird-camera
```

### Check if service is enabled
```bash
systemctl is-enabled bluebird-camera
```

### View full service configuration
```bash
systemctl cat bluebird-camera
```

## Uninstallation

```bash
sudo systemctl stop bluebird-camera
sudo systemctl disable bluebird-camera
sudo rm /etc/systemd/system/bluebird-camera.service
sudo systemctl daemon-reload
```

## Environment Variables

If you need to set environment variables, add them to the service file:

```ini
[Service]
Environment="HF_HUB_OFFLINE=1"
Environment="TRANSFORMERS_OFFLINE=1"
```

## Viewing Web UI

Once the service is running, access the web interface at:
```
http://<pi-ip>:8000
```
