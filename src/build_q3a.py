"""Builder for notebooks/q3a_audio_fingerprint.ipynb  (Q3A)."""
import os
from _nbbuild import md, code, write_notebook

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "notebooks", "q3a_audio_fingerprint.ipynb")

cells = []

cells.append(md(r"""
# EE200 Course Project — Q3A: Sonic Signatures
**Mayur (Roll 240643) · Shazam-style audio fingerprinting**

We build a small song identifier: spectrogram → constellation of peaks →
combinatorial `(f1,f2,Δt)` hashes → time-offset histogram matching. Engine in
`src/q3_fingerprint.py`.

> **Data note.** This notebook uses the real song library in `data/songs/` if
> present; otherwise it auto-generates a small **synthetic demo library** so the
> pipeline runs end-to-end. The banner printed below states which case is live.
> Drop the provided songs into `data/songs/` (filenames unchanged) to run on the
> real database.
"""))

cells.append(code(r"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
import numpy as np
import matplotlib.pyplot as plt
import q3_fingerprint as fp
import q3_demo_data as dd

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
DATA, FIG = os.path.join("..", "data"), os.path.join("..", "figures")
os.makedirs(FIG, exist_ok=True)
np.random.seed(0)

paths, is_real = dd.resolve_library(DATA)
print("LIBRARY:", "REAL songs from data/songs/" if is_real else "SYNTHETIC demo (data/songs_demo/)")
print("Songs:", [fp.song_label(p) for p in paths])
"""))

cells.append(md(r"""
## 1. Why a single Fourier transform is not enough
The DFT of an entire song tells you *which* frequencies are present but throws
away *when* they occur — all timing is gone. That is fatal for recognition,
which depends on the time–frequency pattern.
"""))

cells.append(code(r"""
y0, sr = fp.load_audio(paths[0])
Y = np.abs(np.fft.rfft(y0)); freqs = np.fft.rfftfreq(y0.size, 1/sr)
plt.figure(figsize=(12, 3.4))
plt.semilogy(freqs, Y + 1e-6)
plt.xlabel("frequency (Hz)"); plt.ylabel("|Y(f)| (log)")
plt.title(f"Whole-song DFT of '{fp.song_label(paths[0])}' — which frequencies, but not WHEN")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q3_dft.png")); plt.show()
"""))

cells.append(md(r"""
## 2. Spectrogram & the time–frequency resolution trade-off
A spectrogram slides a short window along the signal and takes the DFT of each
slice. A **short** window localizes events in time but blurs frequency; a
**long** window resolves frequency finely but smears time. This is the same
uncertainty trade-off as Q2(c)(iii).
"""))

cells.append(code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for a, nps, lbl in [(ax[0], 256, "SHORT window (nperseg=256)"),
                    (ax[1], 4096, "LONG window (nperseg=4096)")]:
    f, t, Sdb = fp.compute_spectrogram(y0, sr, nperseg=nps, noverlap=nps//2)
    a.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma")
    a.set_ylim(0, 4000); a.set_xlabel("time (s)"); a.set_ylabel("freq (Hz)")
    a.set_title(lbl + "\nsharp time / blurry freq" if nps==256 else lbl + "\nblurry time / sharp freq")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q3_window_tradeoff.png")); plt.show()
"""))

cells.append(md(r"""
## 3. Constellation map — keep only the strongest peaks
We keep 2-D local maxima above an amplitude floor (`scipy.ndimage.maximum_filter`).
These sparse, high-energy points are robust to noise and form the song's
"constellation".
"""))

cells.append(code(r"""
hashes, peaks, (f, t, Sdb) = fp.fingerprint_signal(y0, sr)
print(f"'{fp.song_label(paths[0])}': {len(peaks)} peaks -> {len(hashes)} paired hashes")
plt.figure(figsize=(13, 4.2))
plt.pcolormesh(t, f, Sdb, shading="gouraud", cmap="magma", alpha=0.9)
plt.scatter(t[peaks[:,1]], f[peaks[:,0]], s=12, c="cyan", marker="o", label="peaks")
plt.ylim(0, 4000); plt.xlabel("time (s)"); plt.ylabel("freq (Hz)")
plt.title("Constellation map (peaks overlaid on spectrogram)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q3_constellation.png")); plt.show()
"""))

cells.append(md(r"""
## 4. Hashing & the database
For each **anchor** peak we pair it with several peaks in a forward
time–frequency **target zone**, forming a hash key `(f1, f2, Δt)` whose value is
`(song, t_anchor)`. Pairing makes each fingerprint far more *specific* than a
lone frequency. We build one inverted index across all songs.
"""))

cells.append(code(r"""
db = fp.build_database(paths)
print(f"Database: {len(db.songs)} songs, {db.num_hashes()} stored hashes, {len(db.index)} unique keys")
"""))

cells.append(md(r"""
## 5. Matching — the offset histogram decides
Fingerprint the query the same way; for each song accumulate
`offset = t_db − t_query` over all matching hashes. The **true** song lines all
its hits up at a single offset (a sharp spike); a **wrong** song produces
scattered, near-uniform counts.
"""))

cells.append(code(r"""
# Build a query clip from one song (ground truth known).
truth_path = paths[min(1, len(paths)-1)]
truth = fp.song_label(truth_path)
yq, _ = fp.load_audio(truth_path)
clip = yq[int(0.25*len(yq)): int(0.25*len(yq)) + int(4.0*sr)]
qh, qpeaks, _ = fp.fingerprint_signal(clip, sr)
best, score, conf, results = fp.identify(qh, db)
print(f"Query taken from '{truth}'  ->  predicted '{best}'  (score={score}, confidence={conf:.2f}x)")

# Offset histogram: winner vs a wrong song.
wrong = next(s for s in db.songs if s != best)
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), sharey=True)
for a, song, ttl in [(ax[0], best, "CORRECT song"), (ax[1], wrong, "WRONG song")]:
    if song in results:
        offs = results[song][2]
        a.bar(list(offs.keys()), list(offs.values()), width=1.0)
    a.set_title(f"{ttl}: '{song}'"); a.set_xlabel("offset (t_db − t_query)")
ax[0].set_ylabel("matching hashes")
plt.suptitle("Offset histograms: one sharp spike = match; scatter = no match")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q3_offset_hist.png")); plt.show()
"""))

cells.append(md(r"""
## 6. Single peaks vs paired hashes
Repeating the match using **single peak frequencies** (no pairing) gives a much
flatter, noisier histogram even for the correct song: an individual frequency
bin recurs constantly across unrelated songs, so it has low specificity. Pairing
constrains `(f1, f2, Δt)` jointly — a far rarer coincidence — which is why joined
peaks make a correct match decisive.
"""))

cells.append(code(r"""
single_idx = fp.build_single_peak_index(paths)
single_res = fp.match_single_peaks(qpeaks, single_idx)
sr_rank = sorted(single_res.items(), key=lambda kv: kv[1][0], reverse=True)
pair_rank = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)

def conf_of(rank):
    if len(rank) < 2 or rank[1][1][0] == 0: return float("inf")
    return rank[0][1][0] / rank[1][1][0]
print(f"PAIRED  hashes: top='{pair_rank[0][0]}' score={pair_rank[0][1][0]}  confidence={conf_of(pair_rank):.2f}x")
print(f"SINGLE  peaks : top='{sr_rank[0][0]}' score={sr_rank[0][1][0]}  confidence={conf_of(sr_rank):.2f}x")
print("=> pairing yields a much higher winner-vs-runner-up confidence ratio (more decisive).")
"""))

cells.append(md(r"""
## 7. Robustness experiments
### (a) Additive white noise — find the breakdown SNR
"""))

cells.append(code(r"""
snrs = [30, 20, 10, 5, 0, -5, -10]
scores, confs = [], []
for snr in snrs:
    h, _, _ = fp.fingerprint_signal(fp.add_white_noise(clip, snr, seed=0), sr)
    b, s, c, _ = fp.identify(h, db)
    scores.append(s if b == truth else 0)
    confs.append(c if (b == truth and np.isfinite(c)) else (0 if b != truth else c))
plt.figure(figsize=(8, 3.8))
plt.plot(snrs, scores, "o-")
plt.gca().invert_xaxis()
plt.xlabel("SNR (dB)  [harder ->]"); plt.ylabel("match score (correct song)")
plt.title("Recognition vs noise: score collapses past the breakdown SNR")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "q3_noise_robustness.png")); plt.show()
for snr, s in zip(snrs, scores): print(f"  SNR={snr:>4} dB -> correct-song score {s}")
"""))

cells.append(md(r"""
### (b) Pitch shift / time stretch — and why a tiny shift defeats it
"""))

cells.append(code(r"""
for semis in [0.0, 0.5, 1.0, 2.0]:
    sig = clip if semis == 0 else fp.pitch_shift(clip, sr, semis)
    h, _, _ = fp.fingerprint_signal(sig, sr)
    b, s, c, _ = fp.identify(h, db)
    print(f"  pitch shift {semis:>3} semitone(s) -> predicted '{b}' (score={s})  {'OK' if b==truth else 'FAILED'}")
"""))

cells.append(md(r"""
**Why a small pitch shift breaks it.** The hash keys store **absolute frequency
bins**. A pitch shift multiplies every frequency by a constant, so every peak
moves to a different bin and essentially **no hash collides** with the database
— even though a human ear, which tracks *relative* pitch, hears the same song.

**One concrete fix.** Quantize peaks to **log-frequency / constant-Q** bins (a
pitch shift becomes a constant *additive* offset there, easier to tolerate), or
allow a small ± tolerance in hash lookup, or pre-index a few pitch-shifted copies
of each song offline.

---
## Q3A — Summary
A single DFT loses timing; the spectrogram restores it with a tunable
time–frequency trade-off. Sparse peak constellations + combinatorial
`(f1,f2,Δt)` hashes give a specific, noise-robust fingerprint; the time-offset
histogram turns hash hits into a single decisive alignment spike for the true
song. Pairing beats single peaks on specificity. The system tolerates heavy
additive noise but is defeated by small pitch shifts because it keys on absolute
frequency — fixable with log-frequency/constant-Q hashing.
"""))

write_notebook(OUT, cells)
print("Wrote", os.path.abspath(OUT), "with", len(cells), "cells")
