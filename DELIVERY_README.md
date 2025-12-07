# Delivery README — Face Detection & Anonymization Pipeline

This document summarizes how to deploy the face detection and anonymization
pipeline on client infrastructure. It includes instructions for Linux and
Windows (Docker and native), offline delivery (OCI tarball), and operational
notes.

CONTENTS
- Overview
- What we deliver
- Quick start: Linux (Docker)
- Quick start: Windows (Docker Desktop)
- Quick start: Windows (native Python fallback)
- Offline delivery (OCI image tarball)
- systemd & service notes (Linux)
- NSSM & Scheduled Task notes (Windows)
- Security and GDPR notes
- Troubleshooting commands


## Overview
The pipeline watches a folder for incoming images, runs face detection,
writes coordinates to EXIF, then anonymizes images (blurring). The
implementation is filesystem-based (no network exposure required) and
is designed to run as a container or as a native Python service.


## What we deliver
- Docker image (tarball: `face-pipeline.tar`) for offline distribution
- `docker-compose.yml` for simple deployment
- `deploy.sh` (Linux installer) and `deploy.ps1` (Windows installer)
- `deploy/face-pipeline.service` (systemd unit example)
- `deploy/NSSM_INSTRUCTIONS.md` (Windows service wrapper instructions)
- `README_DOCKER.md` and this `DELIVERY_README.md` (ops documentation)


## Quick start: Linux (Docker)
1. Copy repository to the server (or extract the package into `/srv/face-pipeline`).
2. Build or load the image:

```bash
# If you provide a tarball
docker load -i face-pipeline.tar
# Or build on the host
docker build -t face-pipeline:latest .
```

3. Run with Docker Compose (in repo root):

```bash
docker compose up -d
```

4. Verify:

```bash
docker ps
docker logs face-pipeline
```

5. To run as a systemd service (optional):

```bash
sudo cp deploy/face-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now face-pipeline.service
sudo systemctl status face-pipeline.service
```


## Quick start: Windows (Docker Desktop)
1. Ensure Docker Desktop is installed and WSL2 enabled (or Docker Engine on Windows Server).
2. Load or build the image:

```powershell
docker load -i face-pipeline.tar
# or
docker build -t face-pipeline:latest .
```

3. Run the container with mounts (PowerShell example):

```powershell
docker run -d --name face-pipeline `
  -v C:\srv\face\photo_input:/app/photo_input `
  -v C:\srv\face\photo_detection_output:/app/photo_detection_output `
  -v C:\srv\face\photo_detection_error:/app/photo_detection_error `
  -v C:\srv\face\photo_anonymization_output:/app/photo_anonymization_output `
  -v C:\srv\face\models:/app/models `
  face-pipeline:latest
```

Notes: Ensure Docker Desktop file sharing permits the drive you mount.


## Quick start: Windows (native Python fallback)
1. Install Python 3.11.
2. Run `deploy.ps1` from an elevated PowerShell to create a venv and install requirements:

```powershell
# run as admin
.\deploy.ps1
```

3. Use NSSM (see `deploy/NSSM_INSTRUCTIONS.md`) or Scheduled Tasks to run `venv\Scripts\python.exe src\watcher.py` at startup.


## Offline delivery (OCI image tarball)
We provide `face-pipeline.tar` which can be loaded with:

```bash
docker load -i face-pipeline.tar
```

Then run with `docker run` or `docker compose up -d` as shown above.


## systemd & service notes (Linux)
- The provided unit runs `docker compose up -d --build` in `/srv/face-pipeline`.
- Alternatively, run `docker run` directly and create a systemd unit invoking `docker start`/`stop`.


## NSSM & Scheduled Task notes (Windows)
- Use NSSM to install a Windows service that runs `python venv\Scripts\python.exe src\watcher.py`.
- Alternatively, create a Scheduled Task to run the watcher at system boot.


## Security and GDPR notes
- No external network exposure is required in watch-mode.
- If exposing an HTTP API later, secure it with TLS and firewall rules.
- Implement data retention and secure deletion policies per client GDPR requirements.


## Troubleshooting
- Check service/container status:
  - Linux: `docker ps`, `docker logs face-pipeline`, `systemctl status face-pipeline.service`
  - Windows Docker: `docker ps`, `docker logs face-pipeline`
  - Windows native: check NSSM service logs or Task Scheduler history
- Confirm mounts inside container:
  `docker exec -it face-pipeline ls -la /app/photo_input`


---
If you want, I will now build the OCI tarball (`face-pipeline.tar`) and place it at the repo root for you to transfer (it may be large). I can also produce a small Windows `deploy.ps1` variant that prefers Docker when available and falls back to venv (we added `deploy.ps1`).
