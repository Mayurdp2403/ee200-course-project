"""Q3B — 'Zapptain America': Streamlit audio-identifier app.

Two modes (both required, both must work on the deployed link):
  - Single-clip: upload one query -> spectrogram, constellation, offset
    histogram, predicted song (in that order).
  - Batch: upload many queries -> results.csv with EXACTLY two columns
    `filename,prediction` (prediction = matched song filename without extension).

Reuses the engine in src/q3_fingerprint.py (no duplicated logic). On Streamlit
Cloud it loads a precomputed hash DB (app/fingerprint_db.pkl) if shipped, else
indexes data/songs/ (or a synthetic demo) at startup.
"""
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import q3_fingerprint as fp          # noqa: E402
import q3_demo_data as dd            # noqa: E402

DATA = os.path.join(ROOT, "data")
# Reuse the same indexed DB the web app ships (real 50-song library).
DB_PICKLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static", "fingerprint_db.pkl")

st.set_page_config(page_title="Zapptain America — Song Identifier", layout="wide")


@st.cache_resource(show_spinner="Indexing song database…")
def get_database():
    """Load the precomputed DB if present, else index the library once."""
    if os.path.exists(DB_PICKLE):
        return fp.load_database(DB_PICKLE), "precomputed", True
    paths, is_real = dd.resolve_library(DATA)
    return fp.build_database(paths), ("real" if is_real else "synthetic demo"), is_real


def load_upload(uploaded) -> tuple[np.ndarray, int]:
    """Read a Streamlit UploadedFile into (signal, sr) via a temp file."""
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        return fp.load_audio(tmp_path)
    finally:
        os.unlink(tmp_path)


def identify_clip(y: np.ndarray, sr: int, db):
    """Fingerprint a clip and return (prediction, score, confidence, results, peaks, spec)."""
    hashes, peaks, spec = fp.fingerprint_signal(y, sr)
    best, score, conf, results = fp.identify(hashes, db)
    return best, score, conf, results, peaks, spec


# --------------------------------------------------------------------------- UI
st.title("🎵 Zapptain America — Audio Fingerprint Identifier")
db, kind, is_real = get_database()
st.caption(
    f"Database: **{len(db.songs)} songs**, {db.num_hashes():,} hashes "
    f"({'precomputed' if kind=='precomputed' else kind}). "
    + ("" if is_real else "⚠️ Running on a synthetic demo library — drop the real "
       "songs into `data/songs/` (or ship `app/fingerprint_db.pkl`) for the real DB.")
)

mode = st.sidebar.radio("Mode", ["Single clip", "Batch (results.csv)"])

if mode == "Single clip":
    up = st.file_uploader("Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"])
    if up is not None:
        y, sr = load_upload(up)
        st.audio(up)
        best, score, conf, results, peaks, (f, t, Sdb) = identify_clip(y, sr, db)

        # 1) Spectrogram
        st.subheader("1 · Spectrogram")
        fig1, ax1 = plt.subplots(figsize=(11, 3.5))
        ax1.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma")
        ax1.set_ylim(0, 4000); ax1.set_xlabel("time (s)"); ax1.set_ylabel("freq (Hz)")
        st.pyplot(fig1)

        # 2) Constellation
        st.subheader("2 · Constellation of peaks")
        fig2, ax2 = plt.subplots(figsize=(11, 3.5))
        ax2.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma", alpha=0.85)
        ax2.scatter(t[peaks[:, 1]], f[peaks[:, 0]], s=10, c="cyan")
        ax2.set_ylim(0, 4000); ax2.set_xlabel("time (s)"); ax2.set_ylabel("freq (Hz)")
        st.pyplot(fig2)

        # 3) Offset histogram for the winning song
        st.subheader("3 · Offset histogram (the deciding vote)")
        fig3, ax3 = plt.subplots(figsize=(11, 3.2))
        if best in results:
            offs = results[best][2]
            ax3.bar(list(offs.keys()), list(offs.values()), width=1.0)
        ax3.set_xlabel("offset (t_db − t_query)"); ax3.set_ylabel("matching hashes")
        st.pyplot(fig3)

        # 4) Prediction
        st.subheader("4 · Prediction")
        if best is None:
            st.error("No confident match found.")
        else:
            st.success(f"### 🎯 {best}")
            c = "∞" if not np.isfinite(conf) else f"{conf:.2f}×"
            st.metric("Match score", score, help="Tallest offset-histogram bin")
            st.write(f"Confidence (winner / runner-up): **{c}**"
                     + ("  — clear match" if (not np.isfinite(conf) or conf >= 2) else
                        "  — ⚠️ ambiguous, treat with caution"))

else:  # Batch mode
    st.write("Upload multiple query clips. Output `results.csv` has exactly "
             "`filename,prediction` (auto-graded format).")
    ups = st.file_uploader("Upload query clips", type=["wav", "mp3", "flac", "ogg", "m4a"],
                           accept_multiple_files=True)
    if ups:
        rows = []
        prog = st.progress(0.0)
        for i, up in enumerate(ups):
            y, sr = load_upload(up)
            best, *_ = identify_clip(y, sr, db)
            rows.append({"filename": up.name, "prediction": best if best else ""})
            prog.progress((i + 1) / len(ups))
        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results_df, use_container_width=True)
        st.download_button("⬇️ Download results.csv",
                           results_df.to_csv(index=False).encode("utf-8"),
                           file_name="results.csv", mime="text/csv")
