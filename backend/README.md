# ClipNova Cloud AI backend

This backend keeps the OpenAI API key off the public GitHub Pages website.

## Required Render environment variables

- `OPENAI_API_KEY`: your OpenAI API key
- `CLIPNOVA_ACCESS_CODE`: private code entered in ClipNova
- `CORS_ORIGINS`: `https://shahzainusa02-star.github.io`
- `OPENAI_TRANSCRIPTION_MODEL`: `whisper-1`
- `OPENAI_ANALYSIS_MODEL`: `gpt-4.1-mini`

## Render commands

- Build: `pip install -r backend/requirements.txt`
- Start: `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`

The browser sends only 16 kHz mono WAV audio chunks. The original video stays on the user's device.
