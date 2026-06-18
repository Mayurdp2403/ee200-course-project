"""Q3B web app backend — 'EE200: Audio Fingerprinting'.

A small Flask server that reuses the fingerprint engine in src/q3_fingerprint.py
and renders the demo-style dark visualizations. Endpoints:

  GET  /                 the single-page app
  GET  /api/library      song cards (label, hashes, peaks, thumbnail)
  GET  /api/samples      built-in sample clips for "Try a sample"
  POST /api/identify     identify one clip -> prediction + viz + timings
  POST /api/batch        identify many clips -> rows (filename, prediction)

Run locally:  python app/web/server.py    (http://localhost:8600)
"""
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import q3_fingerprint as fp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
DB_PATH = os.path.join(STATIC, "fingerprint_db.pkl")
SINGLE_PATH = os.path.join(STATIC, "single_peak_index.pkl")

# Demo-style palette.
BG = "#0c1016"
TEAL = "#2dd4bf"
ORANGE = "#f5a623"
GRID = "#1c2530"

app = Flask(__name__, static_folder=STATIC, template_folder=os.path.join(HERE, "templates"))

# Minimum evidence to call a match (else 'none'), mirroring the demo's threshold.
MIN_SCORE = 5
MIN_CONFIDENCE = 1.3

_DB = None
_SINGLE = None


def get_db():
    global _DB
    if _DB is None:
        _DB = fp.load_database(DB_PATH)
    return _DB


def get_single_index():
    global _SINGLE
    if _SINGLE is None:
        import pickle
        with open(SINGLE_PATH, "rb") as f:
            _SINGLE = pickle.load(f)
    return _SINGLE


def score_all_paired(hashes, db):
    """label -> (score, best_offset, offsets) using paired-hash matching."""
    per_song = defaultdict(lambda: defaultdict(int))
    for key, t_q in hashes:
        for (label, t_db) in db.index.get(key, ()):
            per_song[label][t_db - t_q] += 1
    out = {}
    for label, offs in per_song.items():
        bo = max(offs, key=offs.get)
        out[label] = (offs[bo], bo, dict(offs))
    return out


def score_all_single(peaks, single_index):
    """label -> (score, best_offset, offsets) using single-peak matching."""
    per_song = defaultdict(lambda: defaultdict(int))
    for f_bin, t in peaks:
        for (label, t_db) in single_index.get(int(f_bin) // fp.FREQ_QUANT, ()):
            per_song[label][t_db - int(t)] += 1
    out = {}
    for label, offs in per_song.items():
        bo = max(offs, key=offs.get)
        out[label] = (offs[bo], bo, dict(offs))
    return out


def rank(results):
    return sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)


def apply_transforms(y, sr):
    """Apply optional noise/pitch/stretch transforms from the request form."""
    applied = {}
    snr, pitch, stretch = (request.form.get(k) for k in ("noise_snr", "pitch", "stretch"))
    if snr not in (None, ""):
        y = fp.add_white_noise(y, float(snr)); applied["noise_snr"] = float(snr)
    if pitch not in (None, "") and float(pitch) != 0:
        y = fp.pitch_shift(y, sr, float(pitch)); applied["pitch"] = float(pitch)
    if stretch not in (None, "") and float(stretch) != 1:
        y = fp.time_stretch(y, float(stretch)); applied["stretch"] = float(stretch)
    return y, applied


def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#7a8aa0", labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color("#9fb0c8"); ax.yaxis.label.set_color("#9fb0c8")


def _fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_spectrogram(f, t, Sdb):
    fig, ax = plt.subplots(figsize=(8.4, 3.0)); fig.patch.set_facecolor(BG); _style_ax(ax)
    ax.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma")
    ax.set_ylim(0, f.max()); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    return _fig_to_uri(fig)


def render_constellation(t, f, Sdb, peaks):
    fig, ax = plt.subplots(figsize=(8.4, 3.0)); fig.patch.set_facecolor(BG); _style_ax(ax)
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=5, c=ORANGE, alpha=0.85)
    ax.set_xlabel("time (frames)"); ax.set_ylabel("freq bin")
    return _fig_to_uri(fig)


def render_histogram(offsets, best_offset):
    fig, ax = plt.subplots(figsize=(8.4, 3.0)); fig.patch.set_facecolor(BG); _style_ax(ax)
    if offsets:
        ks = list(offsets.keys()); vs = list(offsets.values())
        ax.bar(ks, vs, width=1.4, color=ORANGE)
        ax.annotate(f"{offsets[best_offset]} hashes align",
                    xy=(best_offset, offsets[best_offset]), color=ORANGE, fontsize=9,
                    xytext=(10, -6), textcoords="offset points")
    ax.set_xlabel("time offset (db frame - query frame)"); ax.set_ylabel("# hashes")
    return _fig_to_uri(fig)


def _line_chart(xs, ys, ok, xlabel, ylabel, invert_x=False):
    """Dark line chart; green markers where the match is still correct, red where lost."""
    fig, ax = plt.subplots(figsize=(7.6, 3.0)); fig.patch.set_facecolor(BG); _style_ax(ax)
    ax.plot(xs, ys, color=TEAL, lw=2, zorder=1)
    for x, y, good in zip(xs, ys, ok):
        ax.scatter([x], [y], s=55, zorder=2,
                   color=(TEAL if good else "#ef4444"),
                   edgecolors="#04201c", linewidths=1)
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, alpha=0.4)
    return _fig_to_uri(fig)


def load_clip_from_request():
    """Return (signal, sr, name) from an uploaded file or a named sample."""
    sample = request.form.get("sample")
    if sample:
        path = os.path.join(STATIC, "samples", f"{sample}.wav")
        y, sr = fp.load_audio(path)
        return y, sr, f"{sample}.wav"
    f = request.files["clip"]
    import tempfile
    suffix = os.path.splitext(f.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name); tmp_path = tmp.name
    try:
        y, sr = fp.load_audio(tmp_path)
    finally:
        os.unlink(tmp_path)
    return y, sr, f.filename


def identify_with_timings(y, sr, db, want_viz=True):
    """Run the pipeline with per-stage timings; optionally render the viz images."""
    t0 = time.perf_counter()
    f, t, Sdb = fp.compute_spectrogram(y, sr)
    t1 = time.perf_counter()
    peaks = fp.find_peaks(Sdb)
    t2 = time.perf_counter()
    hashes = fp.generate_hashes(peaks)
    t3 = time.perf_counter()
    results = score_all_paired(hashes, db)
    t4 = time.perf_counter()
    ranked = rank(results)
    t5 = time.perf_counter()

    best, score, conf = None, 0, 0.0
    if ranked:
        best, (score, best_off, _) = ranked[0]
        runner = ranked[1][1][0] if len(ranked) > 1 else 0
        conf = score / runner if runner else float("inf")
    matched = best if (score >= MIN_SCORE and conf >= MIN_CONFIDENCE) else None

    timings = {
        "spectrogram": round((t1 - t0) * 1000),
        "constellation": round((t2 - t1) * 1000),
        "hashing": round((t3 - t2) * 1000),
        "lookup": round((t4 - t3) * 1000),
        "scoring": round((t5 - t4) * 1000),
    }
    timings["total"] = sum(timings.values())

    out = {
        "prediction": matched,
        "raw_best": best,
        "score": int(score),
        "confidence": (None if conf == float("inf") else round(conf, 1)),
        "n_peaks": int(len(peaks)),
        "n_hashes": len(hashes),
        "n_tracks": len(db.songs),
        "spec_shape": [int(Sdb.shape[0]), int(Sdb.shape[1])],
        "best_offset": int(ranked[0][1][1]) if ranked else 0,
        "candidates": [{"label": l, "score": int(s[0])} for l, s in ranked[:6]],
        "timings": timings,
    }
    if want_viz:
        out["spectrogram_img"] = render_spectrogram(f, t, Sdb)
        out["constellation_img"] = render_constellation(t, f, Sdb, peaks)
        out["histogram_img"] = render_histogram(
            ranked[0][1][2] if ranked else {}, ranked[0][1][1] if ranked else 0)
    return out


# --------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/library")
def api_library():
    with open(os.path.join(STATIC, "library_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return jsonify({"songs": meta, "count": len(meta)})


@app.route("/api/samples")
def api_samples():
    path = os.path.join(STATIC, "samples_meta.json")
    if not os.path.exists(path):
        return jsonify({"samples": []})
    with open(path, encoding="utf-8") as f:
        return jsonify({"samples": json.load(f)})


@app.route("/static/samples/<path:fn>")
def serve_sample(fn):
    return send_from_directory(os.path.join(STATIC, "samples"), fn)


@app.route("/api/identify", methods=["POST"])
def api_identify():
    y, sr, name = load_clip_from_request()
    y, applied = apply_transforms(y, sr)
    result = identify_with_timings(y, sr, get_db(), want_viz=True)
    result["query_name"] = name
    result["transforms"] = applied
    return jsonify(result)


@app.route("/api/robustness", methods=["POST"])
def api_robustness():
    """Sweep noise SNR and pitch shift on one clip; chart how the match holds up."""
    db = get_db()
    y, sr, name = load_clip_from_request()
    # Establish the clean target song first.
    base = score_all_paired(fp.generate_hashes(fp.find_peaks(fp.compute_spectrogram(y, sr)[2])), db)
    if not base:
        return jsonify({"error": "no baseline match"}), 400
    target = rank(base)[0][0]

    def score_for(sig, tgt):
        peaks = fp.find_peaks(fp.compute_spectrogram(sig, sr)[2])
        res = score_all_paired(fp.generate_hashes(peaks), db)
        ranked = rank(res)
        top = ranked[0][0] if ranked else None
        return res.get(tgt, (0,))[0], top

    snrs = [40, 30, 20, 10, 5, 0, -5, -10]
    noise_scores, noise_ok = [], []
    for s in snrs:
        sc, top = score_for(fp.add_white_noise(y, s), target)
        noise_scores.append(int(sc)); noise_ok.append(top == target)

    pitches = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    pitch_scores, pitch_ok = [], []
    for p in pitches:
        sig = y if p == 0 else fp.pitch_shift(y, sr, p)
        sc, top = score_for(sig, target)
        pitch_scores.append(int(sc)); pitch_ok.append(top == target)

    return jsonify({
        "target": target,
        "noise": {"snr": snrs, "score": noise_scores, "correct": noise_ok},
        "pitch": {"semitones": pitches, "score": pitch_scores, "correct": pitch_ok},
        "noise_chart": _line_chart(snrs, noise_scores, noise_ok, "SNR (dB)  [harder ->]",
                                   "match score", invert_x=True),
        "pitch_chart": _line_chart(pitches, pitch_scores, pitch_ok, "pitch shift (semitones)",
                                   "match score", invert_x=False),
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Paired-hash vs single-peak matching on the same query (Q3A part 6)."""
    db = get_db()
    y, sr, name = load_clip_from_request()
    y, _ = apply_transforms(y, sr)
    f, t, Sdb = fp.compute_spectrogram(y, sr)
    peaks = fp.find_peaks(Sdb)
    hashes = fp.generate_hashes(peaks)

    paired = rank(score_all_paired(hashes, db))
    single = rank(score_all_single(peaks, get_single_index()))

    def summarize(ranked, kind):
        if not ranked:
            return {"top": None, "score": 0, "confidence": 0, "hist": render_histogram({}, 0)}
        top, (score, off, offs) = ranked[0]
        runner = ranked[1][1][0] if len(ranked) > 1 else 0
        conf = None if runner == 0 else round(score / runner, 1)
        return {"top": top, "score": int(score), "confidence": conf,
                "hist": render_histogram(offs, off)}

    return jsonify({"paired": summarize(paired, "paired"),
                    "single": summarize(single, "single")})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    db = get_db()
    rows = []
    for f in request.files.getlist("clips"):
        import tempfile
        suffix = os.path.splitext(f.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name); tmp_path = tmp.name
        try:
            y, sr = fp.load_audio(tmp_path)
            r = identify_with_timings(y, sr, db, want_viz=False)
        finally:
            os.unlink(tmp_path)
        rows.append({"filename": f.filename, "prediction": r["prediction"] or "none"})
    matched = sum(1 for r in rows if r["prediction"] != "none")
    return jsonify({"rows": rows, "matched": matched, "total": len(rows)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8600))
    app.run(host="0.0.0.0", port=port, debug=False)
