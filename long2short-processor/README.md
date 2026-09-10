# Long2Short cloud processor

This service performs the long-running video work outside the browser. It scans the full recording at low resolution, scores motion and visual quality, selects moments across the timeline, follows detected faces when reframing, and renders one 9:16 MP4 with FFmpeg.

## Run with Docker

```bash
docker build -t long2short-processor .
docker run --restart unless-stopped -p 8080:8080 \
  -e PROCESSOR_SECRET="your-long-random-secret" \
  -v long2short-data:/data/long2short \
  long2short-processor
```

Set the same secret as `PROCESSOR_SECRET` on the Long2Short Site and set `PROCESSOR_URL` to this service's HTTPS address. A persistent volume lets queued jobs resume after a restart.

Recommended starting server: 4 CPU cores, 8 GB RAM, 100 GB SSD, with HTTPS in front of port 8080.

## Run on Render without Docker

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

The Python dependencies include a bundled FFmpeg binary, so the native Render runtime does not need a system FFmpeg installation. Use `WORK_ROOT=/tmp/long2short` for an initial free-plan test. Upgrade compute and attach persistent storage before processing production two-hour uploads.
