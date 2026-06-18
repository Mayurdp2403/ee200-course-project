"""Verify the Q3 fingerprinting engine end-to-end on the synthetic demo library."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import q3_fingerprint as fp
import q3_demo_data as dd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
paths, is_real = dd.resolve_library(DATA)
print("Library:", "REAL songs" if is_real else "SYNTHETIC demo", "-> ", [fp.song_label(p) for p in paths])

db = fp.build_database(paths)
print(f"DB built: {len(db.songs)} songs, {db.num_hashes()} hashes, {len(db.index)} unique keys")

# Take a 5s query clip from the middle of song index 1 (ground truth = its label).
truth_path = paths[1]
truth = fp.song_label(truth_path)
y, sr = fp.load_audio(truth_path)
dur = len(y) / sr
clip = y[int(0.25*len(y)): int(0.25*len(y)) + int(4.0*sr)]  # 4s clip from 1/4 in
print(f"Song duration {dur:.1f}s; query clip {len(clip)/sr:.1f}s")

def run(label, sig):
    h, _, _ = fp.fingerprint_signal(sig, sr)
    best, score, conf, _ = fp.identify(h, db)
    ok = "OK" if best == truth else "WRONG"
    print(f"  [{label:22}] -> {best!s:14} score={score:4d} conf={conf:4.2f}  (truth={truth})  {ok}")
    return best

print("Ground-truth song:", truth)
run("clean clip", clip)
run("noise SNR=10dB", fp.add_white_noise(clip, 10))
run("noise SNR=0dB", fp.add_white_noise(clip, 0))
run("noise SNR=-5dB", fp.add_white_noise(clip, -5))
run("pitch +1 semitone", fp.pitch_shift(clip, sr, 1.0))
run("pitch +0.5 semitone", fp.pitch_shift(clip, sr, 0.5))

# Single-peak baseline vs pairs (confidence comparison).
single_idx = fp.build_single_peak_index(paths)
_, qpeaks, _ = fp.fingerprint_signal(clip, sr)
single_res = fp.match_single_peaks(qpeaks, single_idx)
ranked = sorted(single_res.items(), key=lambda kv: kv[1][0], reverse=True)
print("Single-peak top-2:", [(l, s[0]) for l, s in ranked[:2]])
print("done")
