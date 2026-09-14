import json
import os
import re
import secrets
import time
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
ANALYSIS_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4.1-mini")
MAX_AUDIO_BYTES = 24 * 1024 * 1024
ALLOWED_CATEGORIES = {
    "Strong opinion",
    "Shocking fact",
    "Important discussion",
    "Emotional highlight",
    "Controversial statement",
    "Actionable insight",
    "Powerful story",
}
RATE_BUCKETS: dict[str, list[float]] = defaultdict(list)

app = FastAPI(title="ClipNova Cloud AI", version="1.0.0")
origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "https://shahzainusa02-star.github.io,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-ClipNova-Code"],
    allow_credentials=False,
)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str = Field(min_length=1, max_length=3000)


class ViralRequest(BaseModel):
    segments: list[TranscriptSegment] = Field(min_length=1, max_length=8000)
    clip_length: int = Field(default=60, ge=15, le=300)
    count: int = Field(default=5, ge=1, le=15)
    instructions: str = Field(default="", max_length=4000)
    range_start: float = Field(default=0, ge=0)
    range_end: float = Field(gt=0)


def require_access(x_clipnova_code: str | None = Header(default=None)) -> None:
    expected = os.getenv("CLIPNOVA_ACCESS_CODE", "").strip()
    if not expected:
        raise HTTPException(503, "Cloud AI access code is not configured on the server.")
    supplied = (x_clipnova_code or "").strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "The ClipNova cloud access code is incorrect.")


def rate_limit(request: Request, scope: str, maximum: int) -> None:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",", 1)[0].strip() or (request.client.host if request.client else "unknown")
    key = f"{scope}:{ip}"
    now = time.time()
    recent = [stamp for stamp in RATE_BUCKETS[key] if now - stamp < 3600]
    if len(recent) >= maximum:
        raise HTTPException(429, "Cloud AI hourly limit reached. Please wait and try again.")
    recent.append(now)
    RATE_BUCKETS[key] = recent


def openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "OpenAI API key is not configured on the server.")
    return OpenAI(api_key=key, timeout=600.0, max_retries=2)


def as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in ("start", "end", "text")
        if hasattr(value, key)
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "access_code_configured": bool(os.getenv("CLIPNOVA_ACCESS_CODE", "").strip()),
        "transcription_model": TRANSCRIPTION_MODEL,
        "analysis_model": ANALYSIS_MODEL,
    }


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    offset: float = Form(default=0),
    language: str = Form(default="auto"),
    _: None = Header(default=None, alias="X-Unused"),
    x_clipnova_code: str | None = Header(default=None),
) -> dict[str, Any]:
    require_access(x_clipnova_code)
    rate_limit(request, "transcribe", 40)
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    await audio.close()
    if not data:
        raise HTTPException(400, "The uploaded audio chunk is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio chunk is larger than 24 MB.")

    kwargs: dict[str, Any] = {
        "model": TRANSCRIPTION_MODEL,
        "file": (audio.filename or "podcast-part.wav", data, audio.content_type or "audio/wav"),
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    code = language.strip().lower()
    if code and code != "auto":
        kwargs["language"] = code

    try:
        result = openai_client().audio.transcriptions.create(**kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"OpenAI transcription failed: {str(exc)[:260]}") from exc

    raw_segments = getattr(result, "segments", None) or as_dict(result).get("segments") or []
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        item = as_dict(raw)
        text = str(item.get("text") or "").strip()
        start = float(item.get("start") or 0) + max(0, offset)
        end = float(item.get("end") or item.get("start") or 0) + max(0, offset)
        if text and end > start:
            segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return {"segments": segments, "text": " ".join(item["text"] for item in segments)}


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?|\s*```\s*$", "", raw.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        begin, finish = cleaned.find("{"), cleaned.rfind("}")
        if begin < 0 or finish <= begin:
            raise ValueError("AI returned no JSON object.")
        value = json.loads(cleaned[begin : finish + 1])
    if not isinstance(value, dict):
        raise ValueError("AI response was not a JSON object.")
    return value


def fit_clip(start: float, end: float, length: int, low: float, high: float) -> tuple[float, float]:
    available = max(0.1, high - low)
    target = min(float(length), available)
    start = max(low, min(float(start), high - 0.1))
    end = max(start + 0.1, min(float(end), high))
    if end - start < target:
        end = min(high, start + target)
        start = max(low, end - target)
    elif end - start > target:
        end = start + target
    return round(start, 3), round(end, 3)


@app.post("/api/viral-moments")
def viral_moments(
    payload: ViralRequest,
    request: Request,
    x_clipnova_code: str | None = Header(default=None),
) -> dict[str, Any]:
    require_access(x_clipnova_code)
    rate_limit(request, "viral", 12)
    if payload.range_end <= payload.range_start:
        raise HTTPException(400, "Invalid podcast analysis range.")

    transcript = "\n".join(
        f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text.strip()}"
        for segment in payload.segments
        if segment.text.strip()
    )
    if not transcript:
        raise HTTPException(400, "No speech was found in the selected podcast range.")

    rubric = f"""
You are ClipNova's senior podcast editor. Read the COMPLETE timestamped transcript before selecting anything.
Find {payload.count} self-contained moments, each approximately {payload.clip_length} seconds long.

Reward:
- a strong clear opinion or memorable claim
- a genuinely surprising/shocking specific fact
- an important useful discussion or actionable insight
- authentic emotion, confession, tension, humor, or transformation
- a controversial but understandable statement or debate
- a strong spoken hook in the opening seconds
- enough context to understand the clip without watching the full podcast
- a complete thought with a satisfying ending

Reject:
- intros, outros, ads, sponsorships, greetings, filler, small talk, repeated points
- random visual changes or loudness without meaningful speech
- contextless sentence fragments, unfinished thoughts, weak setup, and generic advice
- fabricated claims or timestamps not supported by this transcript

Score every selection from 0 to 100 using:
hook 25, novelty 20, emotion 20, usefulness 15, controversy/discussion value 10, clarity/completeness 10.
Prefer different topics and do not overlap selections.
User guidance: {payload.instructions or "Find the strongest broadly shareable moments."}

Return ONLY valid JSON with this exact top-level shape:
{{"moments":[{{"start":12.3,"end":72.3,"score":94,"category":"Strong opinion","hook":"short on-screen hook","reason":"specific reason this can spread","quote":"short supporting transcript excerpt"}}]}}
Allowed category values: {", ".join(sorted(ALLOWED_CATEGORIES))}.
Use only timestamps present in the transcript. Do not include markdown.
""".strip()

    try:
        response = openai_client().responses.create(
            model=ANALYSIS_MODEL,
            instructions=rubric,
            input=transcript,
            max_output_tokens=6000,
        )
        parsed = parse_json_object(response.output_text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"OpenAI viral analysis failed: {str(exc)[:260]}") from exc

    candidates: list[dict[str, Any]] = []
    for raw in parsed.get("moments", []):
        if not isinstance(raw, dict):
            continue
        try:
            start, end = fit_clip(
                float(raw.get("start")),
                float(raw.get("end")),
                payload.clip_length,
                payload.range_start,
                payload.range_end,
            )
            score = max(0, min(100, int(round(float(raw.get("score", 0))))))
        except (TypeError, ValueError):
            continue
        category = str(raw.get("category") or "Important discussion").strip()
        if category not in ALLOWED_CATEGORIES:
            category = "Important discussion"
        candidates.append(
            {
                "start": start,
                "end": end,
                "score": score,
                "category": category,
                "hook": str(raw.get("hook") or "").strip()[:180],
                "reason": str(raw.get("reason") or "").strip()[:500],
                "quote": str(raw.get("quote") or "").strip()[:300],
            }
        )

    chosen: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: value["score"], reverse=True):
        overlap = any(
            max(0.0, min(item["end"], old["end"]) - max(item["start"], old["start"]))
            > payload.clip_length * 0.25
            for old in chosen
        )
        if not overlap:
            chosen.append(item)
        if len(chosen) >= payload.count:
            break
    if not chosen:
        raise HTTPException(502, "Cloud AI did not return usable viral moments.")
    return {"moments": sorted(chosen, key=lambda value: value["start"])}


