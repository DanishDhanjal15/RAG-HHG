# Deployment

The goal is a public URL that keeps working after you close your laptop.

## The shape of the problem

Three artifacts, three very different lifecycles:

| artifact | size | rebuild cost | where it goes |
|---|---:|---|---|
| source | ~2 MB | instant | git |
| ONNX encoders | ~130 MB | ~15 min | built in the Docker **build stage** |
| index | ~170 MB | **hours on CPU** | Hugging Face **dataset repo**, fetched at boot |

The index is the one that dictates the design. It is too big for a git repo
(GitHub rejects files over 100 MB, and a binary rebuilt on every corpus change
ruins the history), and building it inside the image would make every deploy a
multi-hour job. So it is published once and downloaded on first boot.

---

## 1. Build and publish the index

```bash
vrag ingest          # once
vrag build           # hours on CPU; resumable, so a crash costs one stage

huggingface-cli login                    # or set VRAG_HF_TOKEN in .env
python scripts/publish_index.py --repo-id <your-user>/vrag-index
```

The uploader deliberately skips two things: `.done` build markers (a downloaded
index must not look already-built to a machine that has built nothing) and
`vectors.f32` (the raw fp32 memmap is a build intermediate — FAISS holds its own
quantized copy, and it is the single largest file).

Then point the app at it, in `configs/default.yaml`:

```yaml
remote_index:
  repo_id: <your-user>/vrag-index
```

## 2. Create the Space

1. huggingface.co → **New Space** → SDK **Docker** → hardware **CPU basic** (free).
2. Copy `deploy/README.space.md` to the Space repo as `README.md` — its YAML
   frontmatter is what sets the title, emoji and `app_port: 7860`.
3. Push the repo (source only; the index is fetched, not pushed).
4. Space → **Settings → Variables and secrets** → add `VRAG_SARVAM_API_KEY`.
   Without it the app still runs: the mic is disabled with an explanation and the
   text path works, rather than failing silently at the first click.

First boot takes a few minutes — index download, then ONNX and budget-manager
warm-up. `HEALTHCHECK` has a 300 s `start-period` for exactly this. Later boots
find the index on disk and skip the download.

## 3. Verify the deployment, don't assume it

```bash
curl -s https://<user>-<space>.hf.space/api/health
```

Check specifically:

- `chunks` matches the index you published — a mismatch means it fetched an old revision
- `stt_configured: true` — otherwise the secret did not land
- `reranker: true`, `sparse: true` — otherwise it is running degraded

Then re-run the latency benchmark **against the deployed instance**, because
free-tier CPU is not your laptop and the published numbers should say which
machine they came from:

```bash
python bench/run_latency.py --n 200        # locally, for comparison
curl -s https://<user>-<space>.hf.space/api/metrics | python -m json.tool
```

`/api/metrics` exposes live per-stage percentiles from the running process, so
the deployed instance can be checked against the published table rather than
taken on trust. If `budget_warm` is `false`, the numbers are still cold — the
manager is estimating from config rather than measurement.

---

## Running it locally in Docker

```bash
docker build -t vrag .
docker run -p 7860:7860 \
  -e VRAG_SARVAM_API_KEY=... \
  -v "$(pwd)/data/index:/app/data/index:ro" \
  vrag
```

Mounting the index read-only skips the download entirely.

---

## Fallback: Cloudflare Tunnel

If the Space is unavailable, a tunnel exposes the local server on a public URL:

```bash
vrag serve
cloudflared tunnel --url http://localhost:7860
```

Fine for a demo you are present for. **Not** suitable as the submitted link — it
dies when the laptop sleeps, which is precisely when a judge will open it.

---

## What breaks, and what it looks like

| symptom | cause |
|---|---|
| `chunks: 0` or boot crash | index not fetched — check `remote_index.repo_id` and that the dataset repo is public |
| mic disabled, "key not configured" | `VRAG_SARVAM_API_KEY` missing from Space secrets |
| `reranker: false` | ONNX export missing from the image — check the exporter build stage |
| every answer refused | out-of-domain thresholds uncalibrated for this corpus; run `bench/calibrate_thresholds.py` |
| first requests slow, then fine | expected — `budget_warm: false` until the manager has ~8 samples per stage |
