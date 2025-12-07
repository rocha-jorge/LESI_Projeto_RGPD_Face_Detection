#!/usr/bin/env bash
set -euo pipefail

# Simple installer for the face-pipeline service.
# Usage (on the client host):
#   sudo ./deploy.sh
# This will copy the repo files to /srv/face-pipeline, build the Docker image,
# and start the service using docker compose.

INSTALL_DIR=${INSTALL_DIR:-/srv/face-pipeline}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Installing face-pipeline to ${INSTALL_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Please install Docker Engine first." >&2
  exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is not available. Install compose v1 or v2 (docker compose)." >&2
  exit 1
fi

# Create target directory
sudo mkdir -p "${INSTALL_DIR}"

# Copy files (exclude venv, .git, large local files)
rsync -av --delete \
  --exclude ".git" \
  --exclude "venv_face_detection" \
  --exclude "photo_input" \
  --exclude "photo_detection_output" \
  --exclude "photo_detection_error" \
  --exclude "photo_anonymization_output" \
  --exclude "weights" \
  --exclude "venv" \
  "${REPO_ROOT}/" "${INSTALL_DIR}/"

# Ensure required host directories exist (will be used as mounts)
sudo mkdir -p "${INSTALL_DIR}/photo_input" \
  "${INSTALL_DIR}/photo_detection_output" \
  "${INSTALL_DIR}/photo_detection_error" \
  "${INSTALL_DIR}/photo_anonymization_output" \
  "${INSTALL_DIR}/models"

# Build and start via docker compose (try `docker compose`, fallback to `docker-compose`)
cd "${INSTALL_DIR}"
if docker compose version >/dev/null 2>&1; then
  sudo docker compose up -d --build
else
  sudo docker-compose up -d --build
fi

cat <<'EOF'
Installation complete.

Next steps on the client host:
  - Check status: `docker ps` and `docker logs <container-name>`
  - Enable automatic start (if using systemd unit):
      sudo cp deploy/face-pipeline.service /etc/systemd/system/
      sudo systemctl daemon-reload
      sudo systemctl enable --now face-pipeline.service

Model files will be downloaded automatically on first run into the container's
`/app/weights` then moved into `/app/models` by the application.

If the client needs an offline image, run on a networked machine:
  docker build -t face-pipeline:latest .
  docker save face-pipeline:latest -o face-pipeline.tar
Then transfer `face-pipeline.tar` to the client and run:
  docker load -i face-pipeline.tar

EOF
