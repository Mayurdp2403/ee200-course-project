"""Builder for notebooks/q1_image_recovery.ipynb  (Q1A + Q1B)."""
import os
from _nbbuild import md, code, write_notebook

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "notebooks", "q1_image_recovery.ipynb")

cells = []

cells.append(md(r"""
# EE200 Course Project — Q1: Image Recovery & Edge Detection
**Mayur (Roll 240643) · EE200 Signals, Systems & Networks · IIT Kanpur**

This notebook answers **Q1A — Frequency Forensics ("The Ghost Signal")** and
**Q1B — Digital Detective ("Missing Boundaries")**. Core algorithms live in
`src/q1_filters.py` and are imported here so the methodology is reusable and
testable; this notebook supplies the narrative, derivations and figures.
"""))

cells.append(code(r"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
import numpy as np
import matplotlib.pyplot as plt
import q1_filters as q1

# One consistent plot style across the whole report.
plt.rcParams.update({
    "figure.dpi": 110, "image.cmap": "gray",
    "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 10,
})
DATA = os.path.join("..", "data")
FIG  = os.path.join("..", "figures")
os.makedirs(FIG, exist_ok=True)
np.random.seed(0)  # reproducibility for any randomness below
print("Setup OK. q1_filters loaded from src/.")
"""))

# ----------------------------------------------------------------- Q1A theory
cells.append(md(r"""
---
## Q1A — Frequency-Domain Image Recovery System

### Mathematical foundations
An image $I(x,y)$ of size $M\times N$ is a finite 2-D discrete signal. Its
**2-D Discrete Fourier Transform (DFT)** and inverse are

$$F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y)\,
e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)},\qquad
I(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1} F(u,v)\,
e^{+j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}.$$

The raw DFT places the zero-frequency (DC) term at the corner $(0,0)$ and
replicates high frequencies at the array edges. We apply `fftshift` to move DC
to the **centre**, so the low frequencies (the bulk of the image energy) sit in
the middle and spatial frequency grows radially outward — making periodic-noise
impulses easy to read off.

**Why periodic noise is visible in frequency:** a sinusoidal spatial pattern
$\cos(2\pi(u_0 x/M+v_0 y/N))$ added to the image transforms (by the
modulation/shift property) into a **pair of sharp, symmetric impulses** at
$(\pm u_0,\pm v_0)$. Real image content instead forms a smooth energy falloff
clustered near DC. So the corruption is *spatially* diffuse but *spectrally*
concentrated — exactly why we attack it in the Fourier domain.
"""))

cells.append(code(r"""
ghost = q1.load_grayscale(os.path.join(DATA, "ghost_signal_input.png"))
print("Ghost image:", ghost.shape, "intensity range:", ghost.min(), "to", ghost.max())

F = q1.dft2_centered(ghost)
mag_lin = q1.magnitude_spectrum(F, db=False)
mag_db  = q1.magnitude_spectrum(F, db=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].imshow(ghost); ax[0].set_title("Corrupted input  I(x,y)")
im1 = ax[1].imshow(mag_lin, cmap="inferno")
ax[1].set_title("|F(u,v)|  — LINEAR scale")
im2 = ax[2].imshow(mag_db, cmap="inferno")
ax[2].set_title("20·log10|F|  — dB scale")
for a, lbl in zip(ax, ["spatial", "u (cycles)", "u (cycles)"]):
    a.set_xlabel(lbl)
fig.colorbar(im2, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1a_spectrum.png")); plt.show()
"""))

cells.append(md(r"""
**Why the dB scale is essential.** On the *linear* magnitude plot the single DC
term is so large that everything else is crushed to black — the periodic-noise
spikes are invisible. Applying $20\log_{10}(|F|+\varepsilon)$ compresses the
dynamic range so the isolated impulse pairs of the interference stand out as
bright dots arranged on a regular lattice, away from the central low-frequency
cluster. This is the signature of periodic corruption.
"""))

cells.append(code(r"""
# Locate the interference impulses PROGRAMMATICALLY (not by eyeballing):
# subtract a smoothed background to keep only sharp outliers, exclude a disk
# around DC so genuine low-frequency image content is protected.
peaks = q1.detect_noise_peaks(F, exclude_radius=15, neighborhood=7, n_std=6.0)
print(f"Detected {len(peaks)} interference impulses on a regular lattice.")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.imshow(mag_db, cmap="inferno")
ys, xs = zip(*peaks)
ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="cyan", linewidths=1.3)
ax.set_title(f"Detected periodic-noise impulses (n={len(peaks)})")
ax.set_xlabel("u"); ax.set_ylabel("v")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1a_peaks.png")); plt.show()
"""))

cells.append(md(r"""
The detected impulses lie on a **regular 2-D lattice** away from the DC cluster —
the fingerprint of an additive periodic pattern. Their spacing corresponds to
the fundamental period of the grid corrupting the image; the smooth bright blob
at the centre is the *real* image content and must be preserved.

### Notch filter design — ideal vs. Gaussian vs. Butterworth
We zero (or attenuate) a small neighbourhood around each impulse. Three profiles
trade sharpness against ringing:
- **Ideal** (hard cutoff): $H=0$ inside radius $D_0$, else $1$. Sharpest removal
  but the abrupt cutoff causes **Gibbs ringing** in the recovered image.
- **Gaussian**: $H=1-e^{-D^2/2D_0^2}$ — smooth rolloff, minimal ringing.
- **Butterworth** (order $n$): $H=\big(1+(D_0/D)^{2n}\big)^{-1}$ — tunable.
"""))

cells.append(code(r"""
RADIUS = 6.0  # notch radius in px: wide enough to kill each impulse + its skirt,
              # narrow enough not to bite into nearby legitimate frequencies.
rec_ideal, Fi, mask_i = q1.apply_notch_filter(ghost, peaks, RADIUS, kind="ideal")
rec_gauss, Fg, mask_g = q1.apply_notch_filter(ghost, peaks, RADIUS, kind="gaussian")

# Required 4-panel comparison grid.
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].imshow(ghost);                               ax[0].set_title("Corrupted")
ax[1].imshow(mag_db, cmap="inferno");              ax[1].set_title("Spectrum (dB)")
ax[2].imshow(q1.magnitude_spectrum(Fg, True), cmap="inferno"); ax[2].set_title("Filtered spectrum (dB)")
ax[3].imshow(rec_gauss);                           ax[3].set_title("Recovered (Gaussian notch)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1a_grid.png")); plt.show()
print(">>> The hidden message is now readable in the recovered panel. <<<")
"""))

cells.append(code(r"""
# Ideal vs Gaussian: ringing comparison.
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].imshow(rec_ideal); ax[0].set_title("Ideal notch — note ringing artifacts")
ax[1].imshow(rec_gauss); ax[1].set_title("Gaussian notch — smoother, less ringing")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1a_ringing.png")); plt.show()
"""))

cells.append(md(r"""
### Notch-radius trade-off
Does removing *more* frequencies always help? We sweep the notch radius and
measure how much energy the filter removes. Too small a radius leaves residual
noise; too large a radius starts deleting legitimate image frequencies, blurring
the recovered text. The "sweet spot" removes the impulses and little else.
"""))

cells.append(code(r"""
radii = [1, 2, 4, 6, 10, 16, 24]
fig, ax = plt.subplots(2, len(radii)//2 + 1, figsize=(16, 6))
ax = ax.ravel()
energy_removed = []
total_E = np.sum(np.abs(F)**2)
for i, r in enumerate(radii):
    rec, Ff, _ = q1.apply_notch_filter(ghost, peaks, float(r), kind="gaussian")
    energy_removed.append(100 * (1 - np.sum(np.abs(Ff)**2)/total_E))
    ax[i].imshow(rec); ax[i].set_title(f"radius={r}px"); ax[i].axis("off")
# trade-off curve in the last panel
ax[-1].plot(radii, energy_removed, "o-")
ax[-1].set_xlabel("notch radius (px)"); ax[-1].set_ylabel("% spectral energy removed")
ax[-1].set_title("Trade-off: more removal isn't free"); ax[-1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1a_radius_sweep.png")); plt.show()
for r, e in zip(radii, energy_removed):
    print(f"radius={r:>3}px -> {e:5.2f}% energy removed")
"""))

cells.append(md(r"""
**Observation (radius sweep).** Below ~4 px the impulses are only partially
attenuated and faint grid residue survives. From ~4–8 px the message is crisp.
Beyond ~16 px the notches start overlapping the central low-frequency content,
removing genuine image energy and visibly softening/blurring the text — so
**removing more frequencies eventually destroys useful information**. This is the
core practical lesson of frequency-domain restoration.

### Optional interactive widget (report appendix)
A slider over the notch radius (degrades gracefully to a static note if
`ipywidgets` isn't installed).
"""))

cells.append(code(r"""
try:
    from ipywidgets import interact, FloatSlider
    def _show(radius=6.0):
        rec, _, _ = q1.apply_notch_filter(ghost, peaks, radius, kind="gaussian")
        plt.figure(figsize=(7,3.6)); plt.imshow(rec)
        plt.title(f"Recovered @ radius={radius:.1f}px"); plt.axis("off"); plt.show()
    interact(_show, radius=FloatSlider(min=1, max=24, step=1, value=6))
except Exception as e:
    print("ipywidgets not available — see the static radius sweep above.", e)
"""))

# ----------------------------------------------------------------- Q1B theory
cells.append(md(r"""
---
## Q1B — Sobel Edge Detection ("Missing Boundaries")

### Mathematical foundations
Edges are locations of **rapid intensity change**, i.e. large gradient
magnitude. For a continuous image the gradient is
$\nabla I=\big(\partial I/\partial x,\;\partial I/\partial y\big)$. Discretely we
estimate the two partials with the **Sobel kernels**, which combine a central
difference (derivative) in one axis with a $[1,2,1]$ smoothing in the other:

$$G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\qquad
G_y=G_x^{\!\top}=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}.$$

The horizontal and vertical responses are obtained by 2-D convolution
$g_x=G_x * I,\; g_y=G_y * I$, then

$$\text{magnitude}=\sqrt{g_x^2+g_y^2},\qquad
\theta=\operatorname{atan2}(g_y,g_x).$$

Large magnitude ⇒ a sudden intensity transition ⇒ an edge; $\theta$ gives its
orientation.

**Why noise wrecks gradient edge detection, and why smoothing helps.**
Differentiation is a **high-pass** operation: in frequency its response scales
like $|\omega|$, so it *amplifies* high-frequency content. Noise is broadband
and high-frequency-heavy, so raw differentiation amplifies it into a storm of
spurious edges. A **Gaussian blur is a low-pass** filter (its transform is again
a Gaussian, decaying with $|\omega|$); applying it first suppresses the
high-frequency noise so the derivative responds to true structure. The price is
that very fine/weak edges are also low-frequency-smoothed away — the classic
**noise-vs-detail trade-off**.
"""))

cells.append(code(r"""
edge_img = q1.load_grayscale(os.path.join(DATA, "missing_boundaries_input.avif"))
print("Edge-detection input:", edge_img.shape)

mag, direction, (gx, gy) = q1.sobel_edges(edge_img, use_manual=True)

# Validate the from-scratch convolution against SciPy (should be identical).
mag_scipy, _, _ = q1.sobel_edges(edge_img, use_manual=False)
print("max |manual - scipy| Sobel magnitude =", np.abs(mag - mag_scipy).max(),
      "  (0.0 confirms the manual convolution is correct)")

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax[0,0].imshow(edge_img);            ax[0,0].set_title("Input image")
ax[0,1].imshow(gx);                  ax[0,1].set_title(r"$g_x = G_x * I$  (vertical edges)")
ax[0,2].imshow(gy);                  ax[0,2].set_title(r"$g_y = G_y * I$  (horizontal edges)")
ax[1,0].imshow(mag);                 ax[1,0].set_title(r"Gradient magnitude $\sqrt{g_x^2+g_y^2}$")
im = ax[1,1].imshow(direction, cmap="twilight"); ax[1,1].set_title(r"Direction $\theta$ (rad)")
ax[1,2].imshow(q1.threshold_edges(mag, 60)); ax[1,2].set_title("Thresholded edge map (T=60)")
for a in ax.ravel(): a.axis("off")
fig.colorbar(im, ax=ax[1,1], fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1b_sobel.png")); plt.show()
"""))

cells.append(md(r"""
### Effect of noise (no smoothing)
We inject zero-mean Gaussian noise at three levels and run Sobel **directly**.
Spurious edges proliferate as noise grows — the high-pass derivative amplifies
exactly the high-frequency noise.
"""))

cells.append(code(r"""
noise_levels = [0, 15, 35]  # std of added Gaussian noise (intensity units)
fig, ax = plt.subplots(2, len(noise_levels), figsize=(14, 7))
for i, s in enumerate(noise_levels):
    noisy = q1.add_gaussian_noise(edge_img, s, seed=0) if s > 0 else edge_img
    m, _, _ = q1.sobel_edges(noisy, use_manual=False)
    ax[0,i].imshow(np.clip(noisy,0,255)); ax[0,i].set_title(f"Noisy input  σ={s}")
    ax[1,i].imshow(q1.threshold_edges(m, 60)); ax[1,i].set_title(f"Sobel edges  σ={s}")
    for a in (ax[0,i], ax[1,i]): a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1b_noise.png")); plt.show()
"""))

cells.append(md(r"""
### Effect of pre-smoothing — the noise-vs-detail trade-off
Now we fix a noisy input ($\sigma=35$) and pre-smooth with a Gaussian of
increasing $\sigma$ before Sobel. More smoothing kills false edges but also
erases fine/weak true edges — make the trade-off explicit across $\geq 3$
smoothing levels.
"""))

cells.append(code(r"""
noisy = q1.add_gaussian_noise(edge_img, 35, seed=0)
sigmas = [0.0, 1.0, 2.0, 3.5]  # 0 = no smoothing
fig, ax = plt.subplots(2, len(sigmas), figsize=(16, 7))
for i, sg in enumerate(sigmas):
    pre = noisy if sg == 0 else q1.gaussian_blur(noisy, size=int(2*round(2*sg)+1), sigma=sg)
    m, _, _ = q1.sobel_edges(pre, use_manual=False)
    ax[0,i].imshow(np.clip(pre,0,255)); ax[0,i].set_title(f"Pre-smoothed σ={sg}")
    ax[1,i].imshow(q1.threshold_edges(m, 60)); ax[1,i].set_title(f"Edges after σ={sg}")
    for a in (ax[0,i], ax[1,i]): a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1b_smoothing.png")); plt.show()
"""))

cells.append(md(r"""
**Observation.** With no smoothing ($\sigma=0$) the edge map of the noisy image
is dominated by speckle. Light smoothing ($\sigma\approx1$–$2$) removes most
false edges while keeping the strong building/sculpture boundaries. Heavy
smoothing ($\sigma=3.5$) gives the cleanest noise rejection but visibly thins and
drops weak/fine edges — directly answering *"can weak edges be preserved without
introducing excessive noise?"*: only partly, and there is an inherent trade-off.

### Can weak edges be preserved? Canny (bonus) and bilateral pre-filtering
A single global magnitude threshold forces one compromise everywhere. The
standard fix is **Canny**: Gaussian smoothing → gradient → **non-maximum
suppression** (thins edges to 1 px) → **hysteresis thresholding** (a low and a
high threshold, keeping weak edges only if connected to strong ones). This keeps
faint-but-real edges while rejecting isolated noise. A **bilateral filter** is an
edge-preserving alternative to Gaussian pre-smoothing: it averages only over
nearby pixels of *similar intensity*, so it smooths flat regions but does **not**
blur across strong edges.
"""))

cells.append(code(r"""
import cv2
img_u8 = q1.normalize_to_uint8(edge_img)

# Sobel (our pipeline, light smoothing) vs Canny vs bilateral->Sobel.
sobel_mag, _, _ = q1.sobel_edges(q1.gaussian_blur(edge_img, 5, 1.0), use_manual=False)
canny = cv2.Canny(img_u8, 80, 160)  # low/high hysteresis thresholds
bilat = cv2.bilateralFilter(img_u8, d=9, sigmaColor=50, sigmaSpace=9).astype(float)
sobel_bilat, _, _ = q1.sobel_edges(bilat, use_manual=False)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
ax[0].imshow(img_u8);                         ax[0].set_title("Input")
ax[1].imshow(q1.threshold_edges(sobel_mag,60));ax[1].set_title("Sobel (Gaussian pre-smooth)")
ax[2].imshow(canny);                          ax[2].set_title("Canny (NMS + hysteresis)")
ax[3].imshow(q1.threshold_edges(sobel_bilat,60));ax[3].set_title("Sobel (bilateral pre-smooth)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "q1b_canny_bilateral.png")); plt.show()
"""))

cells.append(md(r"""
**Comparison.** Canny produces clean, thin, well-connected contours and recovers
weak edges that a single Sobel threshold misses — it directly answers the
"preserve weak edges" question. Bilateral pre-smoothing keeps edges sharper than
Gaussian pre-smoothing because it refuses to average across intensity
discontinuities, giving crisper Sobel boundaries in textured regions.

---
## Q1 — Summary of answers
**Q1A.** Periodic corruption appears as isolated symmetric impulse pairs on a
lattice in the (centered, dB-scaled) spectrum, distinct from the smooth
near-DC image energy. Notch-rejecting those impulses and inverting the DFT
recovers the hidden message **"QUIZ 2 ON 7th JULY IN TUTORIAL HOURS"**. Gaussian
notches beat ideal ones (less Gibbs ringing); an intermediate notch radius is
best — too large destroys real content. Frequency-domain restoration like this
is used in comms (carrier/interference removal), remote sensing (de-striping),
surveillance and MRI (RF/periodic-artifact suppression).

**Q1B.** Edges = high gradient magnitude, estimated with the Sobel $G_x,G_y$
kernels via 2-D convolution; magnitude $\sqrt{g_x^2+g_y^2}$ and direction
$\operatorname{atan2}(g_y,g_x)$. Differentiation is high-pass so it amplifies
noise into false edges; a Gaussian (or edge-preserving bilateral) low-pass
pre-filter suppresses noise at the cost of fine detail. Weak edges are best
preserved by Canny's non-maximum suppression + hysteresis thresholding.
"""))

write_notebook(OUT, cells)
print("Wrote", os.path.abspath(OUT), "with", len(cells), "cells")
