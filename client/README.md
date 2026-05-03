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

## Detached Run

From the repo root you can launch both the bot and the Vite dev server in
background processes:

```bash
scripts/start_ui_and_bot.sh start
```

This uses `pnpm` when it is on `PATH`, or `corepack pnpm` otherwise. Helpful
commands:

```bash
scripts/start_ui_and_bot.sh status
scripts/start_ui_and_bot.sh stop
scripts/start_client_dev.sh install
scripts/start_client_dev.sh start
scripts/start_bot.sh start
```

Default URLs and logs:

- client: `http://127.0.0.1:5173`
- bot offer endpoint: `http://127.0.0.1:7860/api/offer`
- client log: `logs/client-dev.log`
- bot stdout log: `logs/bot.stdout.log`
- bot debug log: `logs/bot.log`
