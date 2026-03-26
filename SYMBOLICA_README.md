# Arcgentica

ARC-AGI-3 agent harness built on the [Agentica](https://github.com/symbolica-ai/agentica-server) SDK. It uses an orchestrator that delegates to specialized subagents -- explorers, theorists, testers, and solvers -- to figure out game mechanics from scratch and solve multi-level puzzles without any game-specific prompting.

See [`agents/templates/agentica/IDEA.md`](agents/templates/agentica/IDEA.md) for design philosophy and [`agents/templates/agentica/README.md`](agents/templates/agentica/README.md) for architecture.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Git LFS](https://git-lfs.com/) (for pulling recording logs)
- An [ARC-AGI-3 API key](https://three.arcprize.org/)
- An LLM inference provider -- Anthropic API key, OpenRouter, OpenAI, or use the [Agentica platform](https://github.com/symbolica-ai/agentica-server)

## Setup

```bash
# Install Git LFS (if not already installed):
#   macOS: brew install git-lfs
#   Ubuntu: sudo apt install git-lfs
git lfs install

git clone https://github.com/symbolica-ai/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

cp .env.example .env
# Edit .env and set ARC_API_KEY
```

### Inference setup

**Option A: Agentica platform** -- no local server needed. Add to your `.env`:

```
AGENTICA_API_KEY=your_key_here
```

**Option B: Self-hosted agentica-server** -- clone and run the server locally, pointing at your own inference provider.

The server supports multiple providers:

| Provider | `--inference-token` | `--inference-endpoint` |
|---|---|---|
| **Anthropic** | `$ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1/messages` |
| **OpenAI** | `$OPENAI_API_KEY` | `https://api.openai.com/v1/responses` |
| **OpenRouter** | `$OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1/responses` |

In a separate terminal:

```bash
git clone https://github.com/symbolica-ai/agentica-server.git
cd agentica-server
uv run src/application/main.py --disable-otel \
  --inference-token=$ANTHROPIC_API_KEY \
  --inference-endpoint https://api.anthropic.com/v1/messages \
  --sandbox-mode='no_sandbox' \
  --max-concurrent-invocations 200 \
  --port 2345
```

> `--max-concurrent-invocations` caps how many LLM calls the server handles in parallel.
> In practice the arcgentica agent only has a few agents active at once, so a low value
> is fine for a single game. Increase if you want to run parallel agents, swarm runs,
> or multiple games concurrently.

Then add to your `.env` so the agent connects to it:

```
S_M_BASE_URL=http://localhost:2345
```

See [symbolica-ai/agentica-server](https://github.com/symbolica-ai/agentica-server) for details.

## Running the agent

```bash
# Pick a game:
uv run main.py --agent=arcgentica --game=ft09
uv run main.py --agent=arcgentica --game=ls20
uv run main.py --agent=arcgentica --game=vc33
```

### With the visualizer

The visualizer is a browser-based frontend that streams agent activity, grid states, and action history in real time.

```bash
# Start the frontend (in another terminal):
cd agents/templates/agentica/logging/frontend/
python -m http.server

# Run the agent with VISUALIZE=1:
VISUALIZE=1 uv run main.py --agent=arcgentica --game=vc33
```

Open `http://localhost:8000` in your browser. Drop `VISUALIZE=1` if you don't need the frontend.

See [`agents/templates/agentica/README.md`](agents/templates/agentica/README.md) for architecture and [`agents/templates/agentica/logging/README.md`](agents/templates/agentica/logging/README.md) for logging, replays, and visualizer details.

## Tests

```bash
uv run python -m pytest tests/test_smoke.py -v
```

## Scripts

`scripts/` contains optional automation for cloud/CI runs:

- **`scripts/server.sh`** — starts the agentica-server. Expects `INFERENCE_API_KEY` to be set. Defaults to `../agentica-server` for the server checkout (override with `AGENTICA_SERVER_DIR`).
- **`scripts/run.sh`** — starts the server, runs the agent against one or more games, then cleans up. Usage: `./scripts/run.sh ls20` or `./scripts/run.sh ls20,vc33,ft09`.

These are not required for normal use — see [Running the agent](#running-the-agent) above.

## Logs and Replay

See the official ARC-AGI-3 recording logs in `recordings/` and the visualizer replay logs in `visualizer_logs/` (need Git LFS)

- `ls20`:
  - run ID: `ls20-cb3b57cc/09f5aec7-3fcf-41b3-b079-1ed13319b930`
  - scorecard ID: `b8cf9179-4262-462b-8bcc-409204aa205f`
  - as of commit: `8fd5c5e`
- `vc33`:
  - run ID: `vc33-9851e02b/b432e1b7-39c9-498c-a0b2-df6db4b02855`
  - scorecard ID: `87dccf60-b2b8-4a3a-a7e2-44c9e0b9c745`
  - as of commit: `8fd5c5e`
- `ft09`:
  - run ID: `ft09-9ab2447a/142c3c7f-2a22-46b7-a40e-79209935e674`
  - scorecard ID: `e218993-60de-448b-ad98-dafa624d7be8`
  - as of commit: `8fd5c5e`

New runs (no scorecard ID), as of commit `8391486`:

- `ar25`:
  - run ID: `ar25-e3c63847`
- `bp35`:
  - run ID: `bp35-0a0ad940`
- `cd82`:
  - run ID: `cd82-fb555c5d`
- `cn04`:
  - run ID: `cn04-65d47d14`
- `dc22`:
  - run ID: `dc22-4c9bff3e`
- `ft09`:
  - run ID: `ft09-0d8bbf25`
- `g50t`:
  - run ID: `g50t-5849a774`
- `ka59`:
  - run ID: `ka59-9f096b4a`
- `lf52`:
  - run ID: `lf52-271a04aa`
- `lp85`:
  - run ID: `lp85-305b61c3`
- `ls20`:
  - run ID: `ls20-9607627b`
- `m0r0`:
  - run ID: `m0r0-dadda488`
- `r11l`:
  - run ID: `r11l-aa269680`
- `re86`:
  - run ID: `re86-4e57566e`
- `s5i5`:
  - run ID: `s5i5-a48e4b1d`
- `sb26`:
  - run ID: `sb26-7fbdac44`
- `sc25`:
  - run ID: `sc25-f9b21a2f`
- `sk48`:
  - run ID: `sk48-41055498`
- `sp80`:
  - run ID: `sp80-0ee2d095`
- `su15`:
  - run ID: `su15-4c352900`
- `tn36`:
  - run ID: `tn36-ab4f63cc`
- `tr87`:
  - run ID: `tr87-cd924810`
- `tu93`:
  - run ID: `tu93-2b534c15`
- `vc33`:
  - run ID: `vc33-9851e02b`
- `wa30`:
  - run ID: `wa30-ee6fef47`

### Replaying a visualizer log

```bash
# Start the frontend:
cd agents/templates/agentica/logging/frontend/ && python -m http.server &

# Replay a log:
uv run python -m agents.templates.agentica.logging.replay visualizer_logs/ls20-cb3b57cc_20260217_172701.jsonl
```

Then open `http://localhost:8000`. Use `--speed 4` to speed up or `--no-delay` to skip timing.
