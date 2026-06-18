# Deploying the Q3B web app

The identifier is a **Flask** app (`app/web/server.py`). Streamlit Community Cloud
only runs Streamlit scripts, so deploy this on a general Python host — **Render**
(easiest free tier) or **Hugging Face Spaces** work well. The indexed database
ships in the repo as `app/web/static/fingerprint_db.pkl` (~29 MB) plus the
single-peak index and thumbnails, so the app works immediately — the raw `.mp3`
songs are **not** needed at runtime and should not be pushed.

## What ships vs. what stays local
Ships (needed at runtime, all small):
- `app/web/` (server, templates, static CSS/JS)
- `app/web/static/fingerprint_db.pkl`, `single_peak_index.pkl`
- `app/web/static/library_meta.json`, `samples_meta.json`
- `app/web/static/thumbs/*.png`, `samples/*.wav`
- `src/q3_fingerprint.py` (imported by the server)

Stays local (don't push — large and copyrighted):
- `data/songs/*.mp3`  (the 50-track library; `.gitignore`d)

To rebuild all shipped assets after changing the library:
```
python src/build_web_assets.py
```

## Option A — Render (recommended)
1. Push the repo to GitHub.
2. On render.com → New → Web Service → connect the repo. `render.yaml` is picked
   up automatically (build = `pip install -r app/web/requirements.txt`,
   start = `gunicorn --chdir app/web server:app`).
3. Deploy, wait for the build, open the live URL, test all three tabs.

## Option B — Hugging Face Spaces (Docker)
Create a Space (SDK: Docker) with this `Dockerfile`:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install -r app/web/requirements.txt
EXPOSE 7860
CMD ["gunicorn","--chdir","app/web","server:app","--workers","1","--timeout","120","--bind","0.0.0.0:7860"]
```

## Run locally
```
python app/web/server.py        # http://localhost:8600
```

## Before submitting (rubric: broken link = 0 for Q3B)
- Open the **live** link and verify Library, Identify (upload + sample + mic),
  and Batch (download `results.csv`) all work end-to-end.
- Paste the live link **and** the GitHub source link into `report/Q3_report.pdf`.
