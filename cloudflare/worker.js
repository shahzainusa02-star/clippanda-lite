const DEFAULT_ORIGIN = "https://shahzainusa02-star.github.io";

function cors(request, env) {
  const allowed = env.ALLOWED_ORIGIN || DEFAULT_ORIGIN;
  const origin = request.headers.get("Origin") || "";
  return {
    "Access-Control-Allow-Origin": origin === allowed ? origin : allowed,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,X-ClipNova-Code",
    "Vary": "Origin",
  };
}

function json(data, status, request, env) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...cors(request, env) },
  });
}

function authorized(request, env) {
  const expected = String(env.CLIPNOVA_ACCESS_CODE || "").trim();
  const supplied = String(request.headers.get("X-ClipNova-Code") || "").trim();
  return expected && supplied && supplied === expected;
}

function outputText(response) {
  if (typeof response.output_text === "string") return response.output_text;
  return (response.output || [])
    .flatMap((item) => item.content || [])
    .filter((item) => item.type === "output_text" && typeof item.text === "string")
    .map((item) => item.text)
    .join("");
}

function parseObject(raw) {
  const cleaned = String(raw || "")
    .trim()
    .replace(/^\s*```(?:json)?/i, "")
    .replace(/\s*```\s*$/i, "")
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const begin = cleaned.indexOf("{");
    const finish = cleaned.lastIndexOf("}");
    if (begin < 0 || finish <= begin) throw new Error("AI returned no JSON object.");
    return JSON.parse(cleaned.slice(begin, finish + 1));
  }
}

function fitClip(start, end, length, low, high) {
  const available = Math.max(0.1, high - low);
  const target = Math.min(length, available);
  let a = Math.max(low, Math.min(Number(start), high - 0.1));
  let b = Math.max(a + 0.1, Math.min(Number(end), high));
  if (b - a < target) {
    b = Math.min(high, a + target);
    a = Math.max(low, b - target);
  } else if (b - a > target) {
    b = a + target;
  }
  return [Math.round(a * 1000) / 1000, Math.round(b * 1000) / 1000];
}

async function openAI(path, init, env) {
  if (!env.OPENAI_API_KEY) throw new Error("OpenAI API key is not configured.");
  const response = await fetch("https://api.openai.com/v1" + path, {
    ...init,
    headers: { ...(init.headers || {}), Authorization: "Bearer " + env.OPENAI_API_KEY },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body?.error?.message || "OpenAI request failed (" + response.status + ").");
  }
  return body;
}

async function transcribe(request, env) {
  const incoming = await request.formData();
  const audio = incoming.get("audio");
  if (!(audio instanceof File) || !audio.size) return json({ detail: "Audio chunk is empty." }, 400, request, env);
  if (audio.size > 24 * 1024 * 1024) return json({ detail: "Audio chunk is larger than 24 MB." }, 413, request, env);
  const offset = Math.max(0, Number(incoming.get("offset")) || 0);
  const language = String(incoming.get("language") || "auto").toLowerCase();

  const form = new FormData();
  form.append("file", audio, audio.name || "podcast-part.wav");
  form.append("model", env.OPENAI_TRANSCRIPTION_MODEL || "whisper-1");
  form.append("response_format", "verbose_json");
  form.append("timestamp_granularities[]", "segment");
  if (language && language !== "auto") form.append("language", language);

  const result = await openAI("/audio/transcriptions", { method: "POST", body: form }, env);
  const segments = (result.segments || [])
    .map((part) => ({
      start: Math.round((Number(part.start || 0) + offset) * 1000) / 1000,
      end: Math.round((Number(part.end || part.start || 0) + offset) * 1000) / 1000,
      text: String(part.text || "").trim(),
    }))
    .filter((part) => part.text && part.end > part.start);
  return json({ segments, text: segments.map((part) => part.text).join(" ") }, 200, request, env);
}

async function viralMoments(request, env) {
  const payload = await request.json();
  const count = Math.max(1, Math.min(15, Number(payload.count) || 5));
  const clipLength = Math.max(15, Math.min(300, Number(payload.clip_length) || 60));
  const rangeStart = Math.max(0, Number(payload.range_start) || 0);
  const rangeEnd = Number(payload.range_end) || 0;
  const segments = Array.isArray(payload.segments) ? payload.segments.slice(0, 8000) : [];
  if (!segments.length || rangeEnd <= rangeStart) return json({ detail: "No valid transcript range was received." }, 400, request, env);

  const transcript = segments
    .filter((part) => String(part.text || "").trim())
    .map((part) => `[${Number(part.start).toFixed(2)}-${Number(part.end).toFixed(2)}] ${String(part.text).trim()}`)
    .join("\n");
  const categories = [
    "Strong opinion",
    "Shocking fact",
    "Important discussion",
    "Emotional highlight",
    "Controversial statement",
    "Actionable insight",
    "Powerful story",
  ];
  const instructions = `You are ClipNova's senior podcast editor. Read the COMPLETE timestamped transcript before selecting anything.
Find ${count} self-contained moments, each approximately ${clipLength} seconds long.

Reward strong opinions, surprising specific facts, important useful discussions, actionable insights, authentic emotion or tension, controversial but understandable statements, powerful stories, a spoken hook in the opening seconds, enough standalone context, and a complete ending.
Reject intros, outros, ads, greetings, filler, small talk, repetition, random visual changes, loudness without meaningful speech, contextless fragments, generic advice, weak setup, and unfinished thoughts.
Score each moment: hook 25, novelty 20, emotion 20, usefulness 15, controversy/discussion value 10, clarity/completeness 10.
Use different topics, avoid overlap, and use only timestamps and claims supported by the transcript.
User guidance: ${String(payload.instructions || "Find the strongest broadly shareable moments.").slice(0, 4000)}

Return ONLY valid JSON:
{"moments":[{"start":12.3,"end":72.3,"score":94,"category":"Strong opinion","hook":"short on-screen hook","reason":"specific reason this can spread","quote":"short supporting transcript excerpt"}]}
Allowed categories: ${categories.join(", ")}. No markdown.`;

  const ai = await openAI(
    "/responses",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: env.OPENAI_ANALYSIS_MODEL || "gpt-4.1-mini",
        instructions,
        input: transcript,
        max_output_tokens: 6000,
      }),
    },
    env,
  );
  const parsed = parseObject(outputText(ai));
  const candidates = (Array.isArray(parsed.moments) ? parsed.moments : [])
    .map((item) => {
      const [start, end] = fitClip(item.start, item.end, clipLength, rangeStart, rangeEnd);
      const category = categories.includes(item.category) ? item.category : "Important discussion";
      return {
        start,
        end,
        score: Math.max(0, Math.min(100, Math.round(Number(item.score) || 0))),
        category,
        hook: String(item.hook || "").trim().slice(0, 180),
        reason: String(item.reason || "").trim().slice(0, 500),
        quote: String(item.quote || "").trim().slice(0, 300),
      };
    })
    .filter((item) => Number.isFinite(item.start) && Number.isFinite(item.end) && item.end > item.start)
    .sort((a, b) => b.score - a.score);

  const chosen = [];
  for (const item of candidates) {
    const overlaps = chosen.some(
      (old) => Math.max(0, Math.min(item.end, old.end) - Math.max(item.start, old.start)) > clipLength * 0.25,
    );
    if (!overlaps) chosen.push(item);
    if (chosen.length >= count) break;
  }
  if (!chosen.length) return json({ detail: "Cloud AI did not return usable viral moments." }, 502, request, env);
  chosen.sort((a, b) => a.start - b.start);
  return json({ moments: chosen }, 200, request, env);
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(request, env) });
    const url = new URL(request.url);
    if (url.pathname === "/health" && request.method === "GET") {
      return json(
        {
          ok: true,
          openai_configured: Boolean(env.OPENAI_API_KEY),
          access_code_configured: Boolean(env.CLIPNOVA_ACCESS_CODE),
        },
        200,
        request,
        env,
      );
    }
    if (!authorized(request, env)) return json({ detail: "The ClipNova cloud access code is incorrect." }, 401, request, env);
    try {
      if (url.pathname === "/api/transcribe" && request.method === "POST") return await transcribe(request, env);
      if (url.pathname === "/api/viral-moments" && request.method === "POST") return await viralMoments(request, env);
      return json({ detail: "Not found." }, 404, request, env);
    } catch (error) {
      return json({ detail: String(error?.message || error).slice(0, 500) }, 502, request, env);
    }
  },
};
