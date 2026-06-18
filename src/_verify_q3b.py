"""Verify Q3B batch logic: make query clips, run identify, emit results.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, soundfile as sf
import q3_fingerprint as fp
import q3_demo_data as dd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
QDIR = os.path.join(DATA, "queries")
os.makedirs(QDIR, exist_ok=True)

paths, is_real = dd.resolve_library(DATA)
db = fp.build_database(paths)
fp.save_database(db, os.path.join(ROOT, "app", "fingerprint_db.pkl"))
print("DB:", len(db.songs), "songs;", db.num_hashes(), "hashes; saved pickle.")

# Make one query clip per song (4s, taken 1/4 of the way in) -> known ground truth.
truth = {}
for p in paths:
    y, sr = fp.load_audio(p)
    clip = y[int(0.25*len(y)): int(0.25*len(y)) + int(4.0*sr)]
    qname = f"query_{fp.song_label(p)}.wav"
    sf.write(os.path.join(QDIR, qname), clip, sr)
    truth[qname] = fp.song_label(p)

# Batch identify (mirrors the app's batch mode).
rows = []
for qname, gt in truth.items():
    y, sr = fp.load_audio(os.path.join(QDIR, qname))
    best, score, conf, _ = fp.identify(fp.fingerprint_signal(y, sr)[0], db)
    rows.append({"filename": qname, "prediction": best or ""})
df = pd.DataFrame(rows, columns=["filename", "prediction"])
df.to_csv(os.path.join(ROOT, "results.csv"), index=False)

correct = sum(df.loc[i, "prediction"] == truth[df.loc[i, "filename"]] for i in range(len(df)))
print(df.to_string(index=False))
print(f"Batch accuracy: {correct}/{len(df)} correct")
print("results.csv columns:", list(df.columns), "(must be exactly ['filename','prediction'])")
