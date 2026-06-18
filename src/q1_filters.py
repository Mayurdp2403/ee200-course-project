"""Q1 — Frequency-domain image recovery (Q1A) and Sobel edge detection (Q1B).

This module holds the *reusable* core algorithms for Q1 so the notebook can stay
focused on narrative, plots and answering the sub-questions. Everything here is
implemented from foundational NumPy/SciPy operations (the 2D DFT pipeline, the
notch filters, and the Sobel convolution) so the methodology can be defended,
not black-boxed.

Author: Mayur (Roll 240643) — EE200 Course Project, IIT Kanpur.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import convolve2d


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def load_grayscale(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array in [0, 255].

    Tries OpenCV first (fast, handles PNG/JPG), then Pillow (with the AVIF
    plugin) and finally imageio. AVIF is needed for the Q1B input, which OpenCV
    cannot decode on a stock Windows build.

    Args:
        path: Path to the image file.

    Returns:
        2D ``float64`` array of intensities in the range [0, 255].

    Raises:
        IOError: If no backend could decode the file.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img.astype(np.float64)

    # Fallback 1: Pillow (+ pillow-avif-plugin registers the AVIF decoder).
    try:
        import PIL.Image as PILImage

        try:
            import pillow_avif  # noqa: F401  (side-effect: registers AVIF)
        except ImportError:
            pass
        pil = PILImage.open(path).convert("L")
        return np.asarray(pil, dtype=np.float64)
    except Exception:
        pass

    # Fallback 2: imageio.
    try:
        import imageio.v3 as iio

        arr = iio.imread(path)
        if arr.ndim == 3:  # RGB(A) -> luma (ITU-R BT.601).
            arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
        return arr.astype(np.float64)
    except Exception as exc:  # pragma: no cover - last resort
        raise IOError(f"Could not decode image: {path}") from exc


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale an array to the [0, 255] uint8 range.

    Args:
        arr: Any real-valued array.

    Returns:
        ``uint8`` array spanning the full display range. A constant input maps
        to all-zeros (avoids divide-by-zero).
    """
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.round(255.0 * (arr - lo) / (hi - lo)).astype(np.uint8)


# ===========================================================================
# Q1A — Frequency-domain periodic-noise removal
# ===========================================================================
def dft2_centered(img: np.ndarray) -> np.ndarray:
    """Return the centered 2D DFT of an image (``fftshift(fft2(img))``)."""
    return np.fft.fftshift(np.fft.fft2(img))


def idft2_centered(spectrum: np.ndarray) -> np.ndarray:
    """Invert :func:`dft2_centered`; returns the real spatial-domain image."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def magnitude_spectrum(spectrum: np.ndarray, db: bool = True) -> np.ndarray:
    """Magnitude of a complex spectrum, optionally in decibels.

    Args:
        spectrum: Complex DFT (typically centered).
        db: If True return ``20*log10(|F| + eps)``; else the raw magnitude.
            The dB view is essential because the DC term dwarfs every periodic
            spike on a linear scale, hiding exactly the components we hunt for.

    Returns:
        Real-valued magnitude (or log-magnitude) array.
    """
    mag = np.abs(spectrum)
    if db:
        return 20.0 * np.log10(mag + 1e-9)
    return mag


def detect_noise_peaks(
    spectrum: np.ndarray,
    exclude_radius: int = 12,
    neighborhood: int = 7,
    n_std: float = 6.0,
) -> list[tuple[int, int]]:
    """Locate periodic-noise impulses in a centered spectrum, programmatically.

    Periodic spatial corruption shows up as a few sharp, isolated, symmetric
    impulse pairs sitting *away* from the low-frequency cluster at the centre.
    We find them by (a) subtracting a smoothed radial background so only sharp
    outliers remain, (b) keeping points that are both a local maximum and far
    above the global noise floor, and (c) ignoring a disk around DC.

    Args:
        spectrum: Centered complex DFT (from :func:`dft2_centered`).
        exclude_radius: Radius (px) around the DC centre to ignore — protects
            genuine low-frequency image content.
        neighborhood: Side length of the local-maximum window.
        n_std: Outlier threshold in standard deviations above the residual mean.

    Returns:
        List of ``(row, col)`` peak coordinates in the centered spectrum.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    mag_db = magnitude_spectrum(spectrum, db=True)
    # Residual = sharp structure left after removing the smooth energy falloff.
    background = gaussian_filter(mag_db, sigma=3.0)
    residual = mag_db - background

    thresh = residual.mean() + n_std * residual.std()
    local_max = maximum_filter(residual, size=neighborhood)
    candidates = (residual == local_max) & (residual > thresh)

    rows, cols = np.indices(spectrum.shape)
    cr, cc = np.array(spectrum.shape) // 2
    dist = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)
    candidates &= dist > exclude_radius

    ys, xs = np.where(candidates)
    return list(zip(ys.tolist(), xs.tolist()))


def build_notch_mask(
    shape: tuple[int, int],
    peaks: list[tuple[int, int]],
    radius: float,
    kind: Literal["ideal", "gaussian", "butterworth"] = "gaussian",
    order: int = 2,
) -> np.ndarray:
    """Build a frequency-domain notch-reject mask in [0, 1].

    Args:
        shape: ``(rows, cols)`` of the spectrum.
        peaks: Centers of the notches (centered-spectrum coordinates).
        radius: Notch radius / characteristic width D0 (px).
        kind: ``ideal`` (hard cutoff — fast but rings via Gibbs), ``gaussian``
            (smooth rolloff — minimal ringing) or ``butterworth`` (tunable).
        order: Butterworth order (steepness); ignored otherwise.

    Returns:
        Float64 mask the same shape as the spectrum; multiply it with the
        centered DFT to suppress the listed components.
    """
    rows, cols = shape
    yy, xx = np.indices((rows, cols))
    mask = np.ones((rows, cols), dtype=np.float64)

    for (py, px) in peaks:
        d = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        if kind == "ideal":
            notch = (d > radius).astype(np.float64)
        elif kind == "gaussian":
            notch = 1.0 - np.exp(-(d ** 2) / (2.0 * radius ** 2))
        elif kind == "butterworth":
            notch = 1.0 / (1.0 + (radius / np.maximum(d, 1e-6)) ** (2 * order))
        else:  # pragma: no cover
            raise ValueError(f"Unknown notch kind: {kind}")
        mask *= notch
    return mask


def apply_notch_filter(
    img: np.ndarray,
    peaks: list[tuple[int, int]],
    radius: float,
    kind: Literal["ideal", "gaussian", "butterworth"] = "gaussian",
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress periodic noise via a notch filter and reconstruct the image.

    Args:
        img: Grayscale image (float).
        peaks: Noise-impulse centers (centered-spectrum coords).
        radius: Notch radius.
        kind: Notch profile, see :func:`build_notch_mask`.
        order: Butterworth order.

    Returns:
        ``(recovered, filtered_spectrum, mask)`` where ``recovered`` is the real
        spatial-domain image after inverse DFT.
    """
    F = dft2_centered(img)
    mask = build_notch_mask(img.shape, peaks, radius, kind=kind, order=order)
    F_filt = F * mask
    recovered = idft2_centered(F_filt)
    return recovered, F_filt, mask


# ===========================================================================
# Q1B — Sobel edge detection
# ===========================================================================
SOBEL_GX: np.ndarray = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_GY: np.ndarray = SOBEL_GX.T.copy()  # vertical kernel is the transpose


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a normalized 2D Gaussian kernel.

    Args:
        size: Odd kernel side length in pixels.
        sigma: Standard deviation (px).

    Returns:
        ``(size, size)`` kernel summing to 1.
    """
    if size % 2 == 0:
        size += 1  # force odd so the kernel has a well-defined centre
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Low-pass (Gaussian) smoothing via reflect-padded convolution."""
    return convolve2d(img, gaussian_kernel(size, sigma), mode="same", boundary="symm")


def convolve2d_manual(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D correlation with reflect padding (no library convolution).

    Demonstrates the underlying operation: slide the kernel over every pixel,
    multiply-and-sum. Reflect ('symmetric') padding avoids spurious edges at the
    borders. Vectorized with stride tricks so it is fast enough for the notebook
    while remaining a from-scratch implementation.

    Args:
        img: 2D input image.
        kernel: 2D kernel (odd dimensions assumed).

    Returns:
        Filtered image, same shape as ``img``.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    # 'symmetric' reflects including the edge sample, matching SciPy's
    # boundary='symm' so the manual path validates exactly against the library.
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="symmetric")

    # Build a (H, W, kh, kw) sliding-window view, then tensor-contract with the
    # kernel. This is mathematically identical to the explicit double loop.
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sobel_edges(
    img: np.ndarray,
    use_manual: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients, magnitude and direction.

    Args:
        img: Grayscale image (float).
        use_manual: If True use :func:`convolve2d_manual`, else SciPy's
            ``convolve2d`` (both give identical results; the manual path proves
            structural understanding, the SciPy path is a cross-check).

    Returns:
        ``(magnitude, direction, (gx, gy))``. ``magnitude`` is normalized to
        [0, 255]; ``direction`` is in radians from ``atan2(gy, gx)``.
    """
    conv = convolve2d_manual if use_manual else (
        lambda a, k: convolve2d(a, k, mode="same", boundary="symm")
    )
    gx = conv(img, SOBEL_GX)
    gy = conv(img, SOBEL_GY)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return normalize_to_uint8(magnitude).astype(np.float64), direction, (gx, gy)


def threshold_edges(magnitude: np.ndarray, thresh: float) -> np.ndarray:
    """Binarize an edge-magnitude map at ``thresh`` (on a [0, 255] scale)."""
    return (magnitude >= thresh).astype(np.uint8) * 255


def add_gaussian_noise(
    img: np.ndarray, sigma: float, seed: int = 0
) -> np.ndarray:
    """Add zero-mean Gaussian noise of std ``sigma`` (seeded for reproducibility)."""
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, sigma, size=img.shape)
