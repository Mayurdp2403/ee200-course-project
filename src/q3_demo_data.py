"""Song-library resolution + a synthetic fallback so Q3 is runnable before the
real song database (Google-Drive link in data/song_database.txt) is downloaded.

`resolve_library()` returns the real songs in data/songs/ if present; otherwise
it synthesizes a small set of distinct tonal "songs" into data/songs_demo/ so the
whole fingerprint/match/robustness pipeline can be exercised and verified. The
report clearly flags which case is in effect.
"""
from __future__ import annotations

import glob
import os

import numpy as np

SR = 11025
AUDIO_EXTS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")


def _tone_song(notes_hz: list[float], sr: int = SR, note_dur: float = 0.6) -> np.ndarray:
    """Synthesize a simple melody: each note = fundamental + 2 harmonics + envelope."""
    out = []
    for f0 in notes_hz:
        n = int(note_dur * sr)
        tt = np.arange(n) / sr
        wave = (np.sin(2 * np.pi * f0 * tt)
                + 0.5 * np.sin(2 * np.pi * 2 * f0 * tt)
                + 0.25 * np.sin(2 * np.pi * 3 * f0 * tt))
        env = np.minimum(1.0, np.minimum(tt / 0.02, (note_dur - tt) / 0.05))
        out.append(wave * np.clip(env, 0, 1))
    y = np.concatenate(out)
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


def synthesize_demo_library(out_dir: str, sr: int = SR) -> list[str]:
    """Write a handful of distinct synthetic songs; return their paths."""
    import soundfile as sf

    os.makedirs(out_dir, exist_ok=True)
    # Distinct note sequences (Hz) -> distinguishable fingerprints.
    songs = {
        "demo_aurora":   [440, 494, 523, 587, 659, 587, 523, 494],
        "demo_canyon":   [330, 392, 440, 392, 330, 294, 330, 392],
        "demo_meridian": [262, 330, 392, 523, 392, 330, 262, 196],
        "demo_zephyr":   [587, 523, 440, 392, 440, 523, 587, 659],
    }
    paths = []
    for name, notes in songs.items():
        p = os.path.join(out_dir, f"{name}.wav")
        # Repeat the melody x3 (~14 s) so clips, target zones and offset
        # histograms have enough material to be meaningful.
        sf.write(p, _tone_song(notes * 3, sr), sr)
        paths.append(p)
    return paths


def list_audio(folder: str) -> list[str]:
    """All audio files in ``folder`` (sorted)."""
    files: list[str] = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def resolve_library(data_dir: str) -> tuple[list[str], bool]:
    """Return ``(song_paths, is_real)``.

    Prefers real songs in ``data_dir/songs``; falls back to a synthesized demo
    library in ``data_dir/songs_demo`` (created on demand) so the pipeline is
    always runnable.
    """
    real_dir = os.path.join(data_dir, "songs")
    real = list_audio(real_dir)
    if real:
        return real, True
    demo_dir = os.path.join(data_dir, "songs_demo")
    demo = list_audio(demo_dir)
    if not demo:
        demo = synthesize_demo_library(demo_dir)
    return demo, False
