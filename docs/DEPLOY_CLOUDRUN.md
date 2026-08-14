# Deploying to Google Cloud Run

The original target was a Hugging Face Space. That is deployed and correct, but
the account was granted **zero** free CPU-Basic quota
(`Quota exceeded for flavor cpu-basic: current=0, limit=0`) — not a
misconfiguration, just HF's free-tier policy. Cloud Run is the free alternative
that actually fits the app's footprint.

## Why Cloud Run and not the other free tiers

The app needs roughly **700 MB of disk and 600–700 MB of RAM**: ONNX encoders
(~170 MB), the FAISS SQ8 index (~170 MB), an mmapped chunk store, and the Python
runtime. Measured, not estimated.

That single number eliminates most of the free tier:

| host | free RAM | verdict |
|---|---|---|
| Render / Koyeb free | 512 MB | too small |
| Fly.io | 256 MB free | too small |
| **Cloud Run** | up to 32 GB, pay-per-use | **fits, and idles at zero cost** |

Cloud Run's free allowance is 2M requests, 360k GB-seconds and 180k vCPU-seconds
per month. A demo that is idle 99% of the time never approaches it.

## Two decisions baked into the deploy script

**The index is not in the image.** It is fetched from the Hugging Face repo at
boot (`vrag/index/fetch.py`). Baking it in would mean pushing ~700 MB from a home
connection on every deploy; fetching it is a datacenter-to-CDN download measured
in seconds. The image stays small and code deploys stay fast.

The published index also omits `vectors.f32` and `chunks/embed.bin` — both are
build-time artifacts nothing on the serve path reads. That is ~550 MB removed
from every cold start.

**`min-instances` stays at 0.** Setting it to 1 removes cold starts, but bills for
a container running 720 h/month against a 50 h free allowance — it would leave
the free tier immediately. Cold start is ~40–60 s (image pull, index download,
model load, warm-up); `--cpu-boost` and a generous startup probe absorb it, and
the UI shows a waking state rather than an error.

## Deploy

Prerequisites: a Google account, and the
[gcloud CLI](https://cloud.google.com/sdk/docs/install). A billing account must be
attached — Cloud Run's free tier requires one on file, but nothing is charged
while you stay inside the allowance.

```bash
gcloud auth login
gcloud projects create vrag-demo-<something-unique>   # or reuse an existing project

export VRAG_SARVAM_API_KEY=...                        # read by the script
bash deploy/cloudrun/deploy.sh vrag-demo-<something-unique>
```

The script enables the required APIs, creates an Artifact Registry repo, puts the
Sarvam key in **Secret Manager** (not an env var — env vars are visible in the
console, in `gcloud run services describe`, and in deploy logs), builds with Cloud
Build, and deploys.

It prints the public URL when it finishes.

## Verify — don't assume

```bash
curl -s https://<service-url>/api/health
```

Check three things specifically:

- `chunks` matches the published index — a mismatch means it fetched a stale revision
- `stt_configured: true` — otherwise the secret did not bind
- `reranker: true`, `sparse: true` — otherwise it is running degraded

Then re-measure latency **against the deployed instance**, because Cloud Run's
2 vCPU is not the development laptop and the published numbers should say which
machine produced them:

```bash
curl -s https://<service-url>/api/metrics | python -m json.tool
```

If `budget_warm` is `false`, the numbers are still cold — the budget manager is
estimating from config rather than measurement until it has ~8 samples per stage.

## Cost control

```bash
gcloud run services update vrag --region=asia-south1 --max-instances=1   # cap fan-out
gcloud run services delete vrag --region=asia-south1                     # tear down
```

`--max-instances=3` and `--concurrency=8` in the script bound the blast radius:
even a traffic spike cannot run away, because Cloud Run refuses beyond the cap
rather than autoscaling into a bill.

## Failure modes

| symptom | cause |
|---|---|
| container fails to start, logs show index error | `remote_index.repo_id` wrong, or the HF repo is private |
| `stt_configured: false` | secret not bound — re-run the IAM binding step |
| first request times out | cold start exceeded the request timeout; retry, it is warm now |
| `PORT` mismatch / connection refused | the image must honour `$PORT`; the Dockerfile `CMD` uses shell form so it expands |
| build fails on torch download | Cloud Build timeout — the script already sets `--timeout=40m` |
