# EE200 Course Project — Signals, Systems & Networks
**Mayur · Roll 240643 · IIT Kanpur · Instructor: Dr. Tushar Sandhan**

Frequency-domain image recovery, Sobel edge detection, ECG arrhythmia detection,
and a Shazam-style audio identifier. Core algorithms live in `src/` and are
imported by the notebooks and the app (no duplicated logic).

## Repository layout
```
ee200-project/
  data/                         input data (copied from the course dataset)
    ghost_signal_input.png        Q1A
    missing_boundaries_input.avif Q1B
    patient_ecg.npy, template.npy Q2  (REAL data)
    song_database.txt             Q3  (Google-Drive link to the song library)
    songs/                        Q3  <-- put the real songs here (see below)
    songs_demo/                   Q3  auto-generated synthetic fallback
    queries/                      Q3  sample query clips (generated)
  notebooks/
    q1_image_recovery.ipynb       Q1A + Q1B
    q2_ecg_arrhythmia.ipynb       Q2 (a)-(h)
    q3a_audio_fingerprint.ipynb   Q3A
  src/
    q1_filters.py                 DFT/notch filters + Sobel (manual convolution)
    q2_ecg.py                     template build, normalized correlation, find_onset
    q3_fingerprint.py             spectrogram, peak-picking, hashing, matching
    q3_demo_data.py               song-library resolver + synthetic fallback
    build_q1.py / build_q2.py / build_q3a.py   notebook builders (reproducible)
    build_db.py                   precompute app/fingerprint_db.pkl
    build_reports.py              generate the report/*.pdf
    _verify_q3.py / _verify_q3b.py  end-to-end smoke tests
  app/web/                        Q3B web app (PRIMARY) — Flask + HTML/CSS/JS
    server.py                     backend reusing src/q3_fingerprint.py
    templates/index.html, static/ frontend, thumbnails, samples, shipped DB
    DEPLOY.md                     Render / HF Spaces deployment guide
  app/streamlit_app.py            Q3B minimal fallback (single-clip + batch)
  report/  Q1_report.pdf  Q2_report.pdf  Q3_report.pdf
  figures/                        all generated figures
  results.csv                     batch output (filename,prediction)
  requirements.txt
```

## Setup
```bash
python -m pip install -r requirements.txt
```
Python 3.11. Key deps: numpy, scipy, matplotlib, opencv, pillow + pillow-avif-plugin
(for the `.avif` input), librosa + soundfile (audio), streamlit, fpdf2 (reports).

## Reproduce everything
```bash
# 1. (re)build and run the notebooks  — they save figures to figures/
cd src
python build_q1.py   && jupyter nbconvert --to notebook --execute --inplace ../notebooks/q1_image_recovery.ipynb
python build_q2.py   && jupyter nbconvert --to notebook --execute --inplace ../notebooks/q2_ecg_arrhythmia.ipynb
python build_q3a.py  && jupyter nbconvert --to notebook --execute --inplace ../notebooks/q3a_audio_fingerprint.ipynb
# 2. build the PDF reports
python build_reports.py
```
Or just open each notebook in Jupyter and **Restart & Run All** — they run clean
top-to-bottom.

## Results at a glance
- **Q1A** recovers the hidden message **"QUIZ 2 ON 7th JULY IN TUTORIAL HOURS"**.
- **Q1B** Sobel magnitude/direction, the noise-vs-detail smoothing trade-off, and
  a Canny/bilateral comparison. Manual convolution validated against SciPy (0.0 error).
- **Q2** 20 s clip, 75 bpm, f0 = 1.25 Hz; arrhythmia onset detected at **t = 9.6 s
  (sample 2400)** — an inverted beat (rho ~= -1). `find_onset` + spectrogram (nperseg=500).
- **Q3A** spectrogram → constellation → (f1,f2,Δt) hashes → offset-histogram match;
  robust to heavy noise, defeated by small pitch shifts (explained + fix proposed).
- **Q3B** Custom web app (`app/web/`) on the real 50-song library: **Library**
  gallery, **Identify** (upload / sample / **live mic**) with pipeline timings,
  match banner, candidates and spectrogram→constellation→alignment-spike, plus
  **extra analysis** — a **Robustness Lab** (noise & pitch sweeps) and a
  **single-peak vs paired-hash** comparison. **Batch** mode emits `results.csv`
  (exact `filename,prediction`). All sample clips identify correctly with
  50×–516× confidence.

## Run the web app (Q3B)
```bash
python app/web/server.py        # http://localhost:8600
```
Indexed DB ships in `app/web/static/`, so it starts instantly. To re-index after
changing the library: `python src/build_web_assets.py`.

## Q3: the song library
The real 50-song library is indexed into `data/songs/` (filenames unchanged — the
filename without extension is the ground-truth label). The web app ships the
**indexed database** (`app/web/static/fingerprint_db.pkl`), so the raw mp3s are
not needed at runtime and are `.gitignore`d (large/copyrighted). If `data/songs/`
is ever empty, the notebook/Streamlit fallback auto-generates a synthetic demo
library (clearly flagged).

## Q3B deployment (required for marks — broken link = 0)
The web app is **Flask**, so deploy on **Render** (free, `render.yaml` included)
or **Hugging Face Spaces** — Streamlit Cloud only hosts Streamlit scripts. Full
steps in `app/web/DEPLOY.md`. In short:
1. Push this repo to GitHub (the shipped DB pickle + thumbnails go with it).
2. Render → New Web Service → connect repo (`render.yaml` auto-detected), deploy.
3. Open the **live link**, test Library / Identify / Batch end-to-end.
4. Paste the live link + GitHub source link into `report/Q3_report.pdf`.

## Deliverables checklist
- [x] Q1 notebook + `report/Q1_report.pdf`
- [x] Q2 notebook + `report/Q2_report.pdf`
- [x] Q3A pipeline/notebook (real 50-song library) + `report/Q3_report.pdf`
- [x] Q3B web app (Library / Identify+mic / Batch) + `results.csv` (exact format)
- [x] Extra analysis: Robustness Lab + single-peak-vs-pairs comparison
- [x] Real song library indexed into `data/songs/`
- [x] All notebooks run top-to-bottom with no errors
- [x] Q3B deployed (Hugging Face Spaces); live + source links in the Q3 report

**Live app:** https://mayur2403-ee200-audio-fingerprinting.hf.space
**Source:** https://github.com/Mayurdp2403/ee200-course-project
