# Face Detection + Anonymization Docker Deployment

This repository contains a small pipeline that detects faces in images and
writes face coordinates to EXIF, then blurs faces to anonymize images.

The recommended delivery for client deployment is a Docker container that
runs a simple filesystem watcher: when new images are placed into the
`photo_input` folder the pipeline is executed and outputs are written to
`photo_detection_output` and `photo_anonymization_output`.

## Quick start (docker)

1. Build the image:

```bash
docker build -t face-pipeline:latest .
```

2. Create host folders and run the container mounting them:

```bash
mkdir -p /srv/face-pipeline/photo_input
mkdir -p /srv/face-pipeline/photo_detection_output
mkdir -p /srv/face-pipeline/photo_detection_error
mkdir -p /srv/face-pipeline/photo_anonymization_output
mkdir -p /srv/face-pipeline/models

docker run -d \
  --name face-pipeline \
  -v /srv/face-pipeline/photo_input:/app/photo_input \
  -v /srv/face-pipeline/photo_detection_output:/app/photo_detection_output \
  -v /srv/face-pipeline/photo_detection_error:/app/photo_detection_error \
  -v /srv/face-pipeline/photo_anonymization_output:/app/photo_anonymization_output \
  -v /srv/face-pipeline/models:/app/models \
  face-pipeline:latest
```

3. Drop image files into `photo_input`. The watcher polls the input directory
   and runs the pipeline when files are detected. Results go to the mounted
   output folders.

## Quick start (docker-compose)

With the supplied `docker-compose.yml` you can run:

```bash
docker compose up -d --build
```

## Configuration

- `POLL_INTERVAL` (env): seconds between input-folder polls (default: 5)
- `PHOTO_INPUT` (env): path to input folder inside container (default: `/app/photo_input`)

## Notes for the client

- No network exposure is required: the container watches mounted folders.
- For faster processing you can provide a GPU-enabled base image and GPU
  runtime (nvidia-container-toolkit); update the `Dockerfile` and
  replace the base image accordingly.
- Models are downloaded to `/app/weights` by the Ultralytics client on first
  run; this repo will move the cache into `/app/models` automatically.

## Operational hints

- Use the host's process manager (systemd) or container orchestration to
  ensure the container restarts on failure.
- Implement retention/cleanup policies on the host folders if required by
  privacy policies (GDPR).


If you want, I can also generate a `systemd` unit file example and a
short `deploy.sh` script to make installation simpler for the client.
