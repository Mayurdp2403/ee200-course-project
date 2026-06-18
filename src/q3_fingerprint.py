"""Q3 — Sonic Signatures: a Shazam-style audio fingerprinting engine.

Pipeline (all reusable by both the notebook and the Streamlit app):
    audio -> spectrogram -> constellation of spectral peaks
          -> combinatorial (anchor, target) hashes -> database
    query -> same hashes -> per-song offset histogram -> best match.

Design choices follow Wang (2003), "An Industrial-Strength Audio Search
Algorithm" (the Shazam paper): peaks give noise robustness, pairing peaks into
(f1, f2, dt) hashes gives high specificity, and the time-offset histogram turns
a pile of hash hits into a single decisive alignment spike for the true song.

Author: Mayur (Roll 240643) — EE200 Course Project, IIT Kanpur.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Fingerprint hyperparameters (each justified inline; no bare magic numbers).
DEFAULT_SR: int = 11025          # downsample rate: music fingerprints live <~5 kHz,
                                 # 11025 Hz keeps that band and quarters the data.
NPERSEG: int = 1024              # STFT window: df = sr/nperseg ~= 10.8 Hz, dt ~= 46 ms.
NOVERLAP: int = 512              # 50% overlap -> smoother peak tracking.
PEAK_NEIGHBORHOOD: int = 20      # local-max window size in spectrogram bins.
MIN_PEAK_DB: float = -55.0       # amplitude floor: ignore peaks buried in the noise.
FAN_VALUE: int = 15              # how many forward target peaks each anchor pairs with.
DT_MIN: int = 1                  # target-zone time gap bounds (in spectrogram frames)...
DT_MAX: int = 40                 # ...~0.05-1.9 s, the rarely-colliding "sweet spot".
FREQ_QUANT: int = 2              # quantize freq bins in the hash (robustness vs specificity).


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_audio(path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at sample rate ``sr``.

    Uses librosa (handles mp3/wav/flac and resampling). Returns the signal and
    the (possibly resampled) rate.
    """
    import librosa

    y, sr_out = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), int(sr_out)


def song_label(path: str) -> str:
    """Ground-truth label = filename without extension (do NOT rename songs)."""
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Spectrogram + constellation
# ---------------------------------------------------------------------------
def compute_spectrogram(
    y: np.ndarray, sr: int = DEFAULT_SR, nperseg: int = NPERSEG, noverlap: int = NOVERLAP
):
    """Return ``(f, t, Sxx_db)`` — the log-power STFT spectrogram."""
    from scipy.signal import spectrogram

    f, t, Sxx = spectrogram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


def find_peaks(
    Sxx_db: np.ndarray,
    neighborhood: int = PEAK_NEIGHBORHOOD,
    min_db: float = MIN_PEAK_DB,
) -> np.ndarray:
    """Find constellation peaks: 2-D local maxima above an amplitude floor.

    A bin is a peak iff it equals the maximum of its ``neighborhood`` window and
    exceeds ``min_db``. These sparse, high-energy points survive noise far better
    than the full spectrogram.

    Args:
        Sxx_db: Log-power spectrogram (freq x time).
        neighborhood: Side length of the local-maximum window (bins).
        min_db: Amplitude floor in dB.

    Returns:
        ``(n, 2)`` int array of ``(freq_bin, time_bin)`` peak coordinates.
    """
    from scipy.ndimage import maximum_filter

    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    peaks_mask = local_max & (Sxx_db > min_db)
    fbins, tbins = np.where(peaks_mask)
    return np.column_stack([fbins, tbins])


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def generate_hashes(
    peaks: np.ndarray,
    fan_value: int = FAN_VALUE,
    dt_min: int = DT_MIN,
    dt_max: int = DT_MAX,
    freq_quant: int = FREQ_QUANT,
) -> list[tuple[tuple[int, int, int], int]]:
    """Pair each anchor peak with forward neighbours into ``(f1,f2,dt)`` hashes.

    Args:
        peaks: ``(n,2)`` array of ``(freq_bin, time_bin)`` from :func:`find_peaks`.
        fan_value: Max number of target peaks paired per anchor.
        dt_min, dt_max: Target-zone time-gap bounds (frames).
        freq_quant: Frequency-bin quantization (coarser = more robust, less unique).

    Returns:
        List of ``(hash_key, anchor_time)`` where ``hash_key=(f1,f2,dt)``.
    """
    # Sort by time so "forward neighbours" is well defined.
    order = np.argsort(peaks[:, 1])
    pts = peaks[order]
    hashes: list[tuple[tuple[int, int, int], int]] = []
    n = len(pts)
    for i in range(n):
        f1, t1 = int(pts[i, 0]), int(pts[i, 1])
        paired = 0
        for j in range(i + 1, n):
            f2, t2 = int(pts[j, 0]), int(pts[j, 1])
            dt = t2 - t1
            if dt < dt_min:
                continue
            if dt > dt_max:
                break  # sorted by time -> no further target in zone
            key = (f1 // freq_quant, f2 // freq_quant, dt)
            hashes.append((key, t1))
            paired += 1
            if paired >= fan_value:
                break
    return hashes


def fingerprint_signal(y: np.ndarray, sr: int = DEFAULT_SR, **kw):
    """Convenience: signal -> (hashes, peaks, (f, t, Sxx_db)) for plotting/indexing."""
    f, t, Sxx_db = compute_spectrogram(y, sr)
    peaks = find_peaks(Sxx_db)
    hashes = generate_hashes(peaks, **kw)
    return hashes, peaks, (f, t, Sxx_db)


# ---------------------------------------------------------------------------
# Database + matching
# ---------------------------------------------------------------------------
@dataclass
class Database:
    """Inverted index: hash_key -> list of (song_label, anchor_time)."""

    index: dict
    songs: list

    def num_hashes(self) -> int:
        return sum(len(v) for v in self.index.values())


def save_database(db: Database, path: str) -> None:
    """Pickle a built database (so the app cold-starts without re-indexing)."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump({"index": db.index, "songs": db.songs}, f)


def load_database(path: str) -> Database:
    """Load a pickled database written by :func:`save_database`."""
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    return Database(index=d["index"], songs=d["songs"])


def build_database(song_paths: list[str], sr: int = DEFAULT_SR) -> Database:
    """Fingerprint every song and build the hash->(/song, t/) inverted index."""
    index: dict = defaultdict(list)
    songs: list = []
    for path in song_paths:
        label = song_label(path)
        songs.append(label)
        y, _ = load_audio(path, sr)
        hashes, _, _ = fingerprint_signal(y, sr)
        for key, t1 in hashes:
            index[key].append((label, t1))
    return Database(index=dict(index), songs=songs)


def match_hashes(query_hashes, db: Database) -> dict:
    """Match query hashes to the DB via per-song time-offset histograms.

    For every query hash that hits the DB, accumulate ``offset = t_db - t_query``
    per song. A true match aligns nearly all hits at one offset (a sharp spike);
    wrong songs scatter. Score = the tallest offset bin.

    Returns:
        ``{song_label: (score, best_offset, offset_counter)}``.
    """
    per_song = defaultdict(lambda: defaultdict(int))
    for key, t_q in query_hashes:
        for (label, t_db) in db.index.get(key, ()):  # noqa: E501
            per_song[label][t_db - t_q] += 1

    results = {}
    for label, offsets in per_song.items():
        best_offset = max(offsets, key=offsets.get)
        results[label] = (offsets[best_offset], best_offset, dict(offsets))
    return results


def identify(query_hashes, db: Database):
    """Return ``(best_label, score, confidence, results)``.

    ``confidence`` = best score / runner-up score (>1 means a clear winner; near
    1 means ambiguous). ``results`` is the full :func:`match_hashes` dict for
    plotting the deciding histogram.
    """
    results = match_hashes(query_hashes, db)
    if not results:
        return None, 0, 0.0, results
    ranked = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    best_label, (best_score, _, _) = ranked[0]
    runner = ranked[1][1][0] if len(ranked) > 1 else 0
    confidence = best_score / runner if runner > 0 else float("inf")
    return best_label, best_score, confidence, results


def match_single_peaks(query_peaks: np.ndarray, db_single: dict) -> dict:
    """Baseline matcher using single peak frequencies only (no pairing).

    Demonstrates why pairing matters: a lone frequency bin is far less specific,
    so the offset histogram stays noisy/flat even for the correct song.
    """
    per_song = defaultdict(lambda: defaultdict(int))
    for f, t in query_peaks:
        for (label, t_db) in db_single.get(int(f) // FREQ_QUANT, ()):
            per_song[label][t_db - int(t)] += 1
    results = {}
    for label, offsets in per_song.items():
        best = max(offsets, key=offsets.get)
        results[label] = (offsets[best], best, dict(offsets))
    return results


def build_single_peak_index(song_paths: list[str], sr: int = DEFAULT_SR) -> dict:
    """Single-peak inverted index: freq_bin -> list of (label, time) (baseline)."""
    index = defaultdict(list)
    for path in song_paths:
        label = song_label(path)
        y, _ = load_audio(path, sr)
        _, peaks, _ = fingerprint_signal(y, sr)
        for f, t in peaks:
            index[int(f) // FREQ_QUANT].append((label, int(t)))
    return dict(index)


# ---------------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------------
def add_white_noise(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB). Seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by ``n_steps`` semitones (librosa), preserving duration."""
    import librosa

    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch by ``rate`` (>1 faster), preserving pitch (librosa)."""
    import librosa

    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
