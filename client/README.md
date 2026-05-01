# Nemotron Voice Client

Custom React client for the local SmallWebRTC bot.

## Run

Start the Pipecat bot on port `7860`, then:

```bash
pnpm i
pnpm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173/`.

The development server proxies `/start`, `/api/*`, and `/sessions/*` to
`http://127.0.0.1:7860` by default. Set `PIPECAT_BOT_URL` to override that:

```bash
PIPECAT_BOT_URL=http://127.0.0.1:7860 pnpm run dev
```
