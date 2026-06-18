"""Precompute the fingerprint database and pickle it to app/fingerprint_db.pkl.

Run this once after placing the real songs in data/songs/ so the deployed app
cold-starts instantly instead of re-indexing audio.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import q3_fingerprint as fp
import q3_demo_data as dd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "app", "fingerprint_db.pkl")

if __name__ == "__main__":
    paths, is_real = dd.resolve_library(DATA)
    print(f"Indexing {len(paths)} {'REAL' if is_real else 'SYNTHETIC demo'} songs:",
          [fp.song_label(p) for p in paths])
    db = fp.build_database(paths)
    fp.save_database(db, OUT)
    print(f"Saved DB -> {OUT}  ({len(db.songs)} songs, {db.num_hashes()} hashes)")
