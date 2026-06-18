"""Index the real song library and precompute all web-app assets.

Outputs (under app/web/):
  static/fingerprint_db.pkl   the hash index used by the server
  static/library_meta.json    [{label, hashes, peaks, duration, thumb, cmap}]
  static/thumbs/<label>.png    constellation thumbnail per song
  static/samples/sampleN.wav   a few query clips cut from random songs
  static/samples_meta.json     [{name, file, truth}]  (ground truth for the demo)
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

import q3_fingerprint as fp
import q3_demo_data as dd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "app", "web", "static")
THUMBS = os.path.join(WEB, "thumbs")
SAMPLES = os.path.join(WEB, "samples")
for d in (THUMBS, SAMPLES):
    os.makedirs(d, exist_ok=True)

# Rotating colormaps so the gallery cards look varied (like the demo).
CMAPS = ["viridis", "plasma", "cool", "spring", "summer", "winter", "autumn",
         "YlGnBu", "PuBuGn", "BuPu", "GnBu", "YlOrRd"]


def render_thumb(peaks, path, cmap):
    """Render a small constellation thumbnail (freq vs time, colored by freq)."""
    fig, ax = plt.subplots(figsize=(3.4, 1.7), dpi=100)
    fig.patch.set_facecolor("#0c1016"); ax.set_facecolor("#0c1016")
    if len(peaks):
        ax.scatter(peaks[:, 1], peaks[:, 0], s=2.0, c=peaks[:, 0], cmap=cmap, alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    paths, is_real = dd.resolve_library(DATA)
    print(f"Indexing {len(paths)} {'REAL' if is_real else 'SYNTHETIC'} songs...")
    from collections import defaultdict

    index = defaultdict(list)
    single_index = defaultdict(list)  # freq_bin -> [(label, t)] for the single-peak baseline
    songs, meta = [], []
    sample_sources = []  # (label, signal, sr) candidates to cut samples from

    for i, path in enumerate(paths):
        label = fp.song_label(path)
        y, sr = fp.load_audio(path)
        hashes, peaks, _ = fp.fingerprint_signal(y, sr)
        for key, t1 in hashes:
            index[key].append((label, t1))
        for pf, pt in peaks:
            single_index[int(pf) // fp.FREQ_QUANT].append((label, int(pt)))
        songs.append(label)
        cmap = CMAPS[i % len(CMAPS)]
        thumb = f"thumbs/{label}.png"
        render_thumb(peaks, os.path.join(WEB, thumb), cmap)
        meta.append({"label": label, "hashes": len(hashes), "peaks": int(len(peaks)),
                     "duration": round(len(y) / sr, 1), "thumb": thumb, "cmap": cmap})
        if 3 <= i < 8:  # collect 5 mid-library songs as sample sources
            sample_sources.append((label, y, sr))
        print(f"  [{i+1:>2}/{len(paths)}] {label:42} {len(hashes):>6} hashes")

    db = fp.Database(index=dict(index), songs=songs)
    fp.save_database(db, os.path.join(WEB, "fingerprint_db.pkl"))
    import pickle
    with open(os.path.join(WEB, "single_peak_index.pkl"), "wb") as f:
        pickle.dump(dict(single_index), f)
    with open(os.path.join(WEB, "library_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Cut sample query clips (~18 s from 1/3 in) for the "Try a sample" feature.
    samples_meta = []
    for n, (label, y, sr) in enumerate(sample_sources, start=1):
        start = int(len(y) / 3)
        clip = y[start:start + int(18 * sr)]
        fname = f"samples/sample{n}.wav"
        sf.write(os.path.join(WEB, fname), clip, sr)
        samples_meta.append({"name": f"sample{n}", "file": fname, "truth": label})
    with open(os.path.join(WEB, "samples_meta.json"), "w", encoding="utf-8") as f:
        json.dump(samples_meta, f, indent=2)

    print(f"\nDB: {len(db.songs)} songs, {db.num_hashes()} hashes.")
    print(f"Thumbnails: {len(meta)}; samples: {len(samples_meta)}")
    print("Wrote web assets to", WEB)


if __name__ == "__main__":
    main()
