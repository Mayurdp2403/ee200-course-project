"""Q2 — Midnight Episode: ECG arrhythmia detection via template correlation.

Core, importable, unit-testable logic for Q2: building the one-beat template,
the normalized cross-correlation score rho(m), the beat-by-beat onset detector
``find_onset`` required in part (g), and a spectrogram helper for part (h).

Sampling/anatomy facts fixed by the problem statement:
    fs = 250 Hz, N = 5000 samples (=> 20 s clip), healthy beat every 0.8 s
    => 200 samples/beat, heart rate 75 bpm, fundamental f0 = 1.25 Hz.

Author: Mayur (Roll 240643) — EE200 Course Project, IIT Kanpur.
"""
from __future__ import annotations

import numpy as np

FS_HZ: float = 250.0  # sampling rate fixed by the problem statement


def normalized_correlation(template: np.ndarray, segment: np.ndarray) -> float:
    """Normalized cross-correlation (cosine similarity) of two equal-length signals.

    Implements ``rho = <t, x> / (||t|| ||x||)``. By Cauchy-Schwarz this lies in
    [-1, 1]; +1 is a perfect shape match, -1 a perfectly inverted one. Dividing
    by the energies makes the score insensitive to amplitude scaling and baseline
    wander, so it measures *shape*, not size — part (d)(ii).

    Args:
        template: Reference beat ``t[k]``.
        segment: Signal segment ``x[m:m+L]`` of the same length.

    Returns:
        Correlation score in [-1, 1]; 0.0 if either signal has zero energy.
    """
    t = np.asarray(template, dtype=np.float64)
    x = np.asarray(segment, dtype=np.float64)
    denom = np.linalg.norm(t) * np.linalg.norm(x)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(t, x) / denom)


def correlation_trace(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Sample-by-sample normalized correlation rho(m) across the whole signal.

    Used for the dense rho-vs-time plot in part (e). Slides the template by one
    sample at a time; positions where a full template would run past the end are
    omitted.

    Args:
        signal: Full recording ``x[n]``.
        template: One-beat template ``t[k]`` of length L.

    Returns:
        Array of length ``len(signal) - L + 1`` with rho at each start index.
    """
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    L = t.size
    t_norm = np.linalg.norm(t)

    windows = np.lib.stride_tricks.sliding_window_view(x, L)
    seg_norms = np.linalg.norm(windows, axis=1)
    dots = windows @ t
    denom = seg_norms * t_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(denom > 1e-12, dots / denom, 0.0)
    return rho


def find_onset(
    ecg_signal: np.ndarray, template: np.ndarray, threshold: float = 0.5
) -> int:
    """Return the start index of the first beat whose shape breaks down.

    Part (g): step through the recording **beat-by-beat** (jumping forward by
    ``len(template)`` each time, not sample-by-sample), score each beat with the
    normalized correlation, and return the start index ``m`` of the first beat
    whose score drops strictly below ``threshold``. Returns -1 if every beat
    stays at or above the threshold.

    Args:
        ecg_signal: The recording ``x[n]``.
        template: One clean healthy-beat template ``t[k]``.
        threshold: Match threshold (default 0.5, as fixed by the question).

    Returns:
        Sample index of the first anomalous beat, or -1 if none is found.
    """
    x = np.asarray(ecg_signal, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    L = t.size
    for m in range(0, x.size - L + 1, L):  # non-overlapping, one stride per beat
        if normalized_correlation(t, x[m : m + L]) < threshold:
            return m
    return -1


def beat_scores(
    ecg_signal: np.ndarray, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-beat start indices and their normalized correlation scores.

    Companion to :func:`find_onset` for plotting/inspection: same non-overlapping
    beat stride, but keeps every score instead of stopping at the first dropout.

    Args:
        ecg_signal: The recording ``x[n]``.
        template: One-beat template.

    Returns:
        ``(starts, scores)`` arrays of equal length.
    """
    x = np.asarray(ecg_signal, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    L = t.size
    starts = np.arange(0, x.size - L + 1, L)
    scores = np.array([normalized_correlation(t, x[m : m + L]) for m in starts])
    return starts, scores


def compute_spectrogram(
    ecg_signal: np.ndarray, fs: float = FS_HZ, nperseg: int = 500
):
    """Spectrogram for part (h) with a justified window length.

    ``nperseg`` sets the frequency resolution ``df = fs/nperseg``. To resolve the
    harmonic spacing f0 = 1.25 Hz we need ``df <= ~1.25 Hz`` => ``nperseg >= 200``.
    We use 500 (df = 0.5 Hz) so the harmonics at f0, 2f0, 3f0, ... sit on
    clean, well-separated horizontal bands in the healthy region.

    Args:
        ecg_signal: The recording (1D).
        fs: Sampling rate (Hz).
        nperseg: STFT window length in samples.

    Returns:
        ``(f, t, Sxx)`` as returned by :func:`scipy.signal.spectrogram`.
    """
    from scipy.signal import spectrogram

    return spectrogram(
        np.asarray(ecg_signal, dtype=np.float64),
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
