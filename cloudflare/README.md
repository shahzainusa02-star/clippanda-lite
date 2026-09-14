# ClipNova Cloudflare Worker

This is the free-hosting alternative for the ClipNova secure AI backend.

Set these secrets in Cloudflare before deployment:

- `OPENAI_API_KEY`
- `CLIPNOVA_ACCESS_CODE`

Non-secret configuration is in the root `wrangler.toml`.

Deploy with Wrangler:

```sh
npx wrangler secret put OPENAI_API_KEY
npx wrangler secret put CLIPNOVA_ACCESS_CODE
npx wrangler deploy
```

The Worker accepts 16 kHz mono WAV chunks, sends them to OpenAI transcription, and asks the analysis model to rank semantically viral podcast moments.
