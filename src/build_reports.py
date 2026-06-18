"""Generate the in-depth per-question PDF reports in report/ using fpdf2.

Each report addresses every sub-question / requirement of the assignment spec
explicitly and at length: full derivations, methodology and justification,
figure-by-figure observations, and discussion. Run the notebooks first so the
figures/ directory is current.
"""
import os

import matplotlib
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
REPORT = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

TTF_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_REG = os.path.join(TTF_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(TTF_DIR, "DejaVuSans-Bold.ttf")
FONT_IT = os.path.join(TTF_DIR, "DejaVuSans-Oblique.ttf")

# This fpdf2/font build renders Latin-1 reliably only; transliterate math symbols.
_SUBS = {
    "ρ": "rho", "σ": "sigma", "√": "sqrt", "∇": "grad", "Σ": "Sum", "Δ": "Delta",
    "ω": "w", "θ": "theta", "π": "pi", "ε": "eps", "μ": "u", "∂": "d", "φ": "phi",
    "λ": "lambda", "τ": "tau", "α": "alpha", "β": "beta", "γ": "gamma",
    "⇒": "=>", "→": "->", "≈": "~=", "≥": ">=", "≤": "<=", "×": "x", "−": "-",
    "·": "*", "²": "^2", "³": "^3", "°": "deg", "•": "-", "—": "--", "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "′": "'", "″": "''", "ᵀ": "^T",
    "‖": "||", "∈": " in ", "±": "+/-", "≲": "<~", "≳": ">~", "₀": "0", "₁": "1",
    "₂": "2", "…": "...", "∝": " proportional to ", "∞": "inf", "≠": "!=",
    "⊛": "*", "∗": "*", "≫": ">>", "≪": "<<", "∫": "integral",
}
_LEFTOVER = set()


def _s(txt):
    for k, v in _SUBS.items():
        txt = txt.replace(k, v)
    out = []
    for ch in txt:
        if ord(ch) < 128:
            out.append(ch)
        else:
            _LEFTOVER.add(ch); out.append("?")
    return "".join(out)


class Report(FPDF):
    def __init__(self, title):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=16)
        self.add_font("DejaVu", "", FONT_REG)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_font("DejaVu", "I", FONT_IT)
        self.add_page()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8); self.set_text_color(140)
        self.cell(0, 6, _s(self.title_text), align="R"); self.ln(7)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("DejaVu", "", 8); self.set_text_color(150)
        self.cell(0, 6, f"page {self.page_no()}", align="C")

    def _mc(self, h, txt, **kw):
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, h, _s(txt), **kw)

    def h1(self, t):
        self.set_font("DejaVu", "B", 15); self.ln(2); self.set_text_color(20, 90, 80)
        self._mc(8, t); self.set_text_color(0); self.ln(1)

    def h2(self, t):
        self.set_font("DejaVu", "B", 12); self.ln(1.5); self._mc(6.5, t); self.ln(0.5)

    def h3(self, t):
        self.set_font("DejaVu", "B", 10.5); self.ln(1); self._mc(5.6, t)

    def body(self, t):
        self.set_font("DejaVu", "", 10); self._mc(5.2, t); self.ln(1.2)

    def eq(self, t):
        """Centered, monospace-ish emphasis line for an equation."""
        self.set_font("DejaVu", "I", 10); self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin); self.multi_cell(self.epw, 5.4, _s(t), align="C")
        self.set_text_color(0); self.ln(1)

    def figure(self, name, caption, w=170):
        path = os.path.join(FIG, name)
        if not os.path.exists(path):
            self.body(f"[figure {name} missing - run the notebook]"); return
        if self.get_y() + w * 0.42 > 275:
            self.add_page()
        self.image(path, x=(210 - w) / 2, w=w)
        self.set_font("DejaVu", "I", 8.6); self.set_text_color(90)
        self._mc(4.3, caption); self.set_text_color(0); self.ln(2)


def title_block(pdf, subtitle):
    pdf.set_font("DejaVu", "B", 19); pdf.set_text_color(15, 80, 72)
    pdf._mc(9, pdf.title_text); pdf.set_text_color(0)
    pdf.set_font("DejaVu", "", 11); pdf.set_text_color(90)
    pdf._mc(6, subtitle); pdf.set_text_color(0)
    pdf.set_draw_color(20, 150, 130); pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
    pdf.ln(5)


# ===========================================================================
# Q1 REPORT
# ===========================================================================
def build_q1():
    p = Report("Q1 - Frequency-Domain Image Recovery & Edge Detection")
    title_block(p, "Q1A 'The Ghost Signal' (3%)  +  Q1B 'Missing Boundaries' (2%)  "
                   "- EE200 Course Project")

    p.body(
        "This report treats an image as a finite two-dimensional discrete signal "
        "I(x,y) and analyses it in the Fourier domain. Q1A removes periodic "
        "interference to recover a hidden message; Q1B estimates spatial gradients "
        "to extract object boundaries. All core operations - the 2-D DFT pipeline, "
        "the notch filters and the Sobel convolution - are implemented from "
        "foundational NumPy/SciPy operations (src/q1_filters.py); no black-box edge "
        "or transform routine is used for the core logic. Only the provided images "
        "(ghost_signal_input.png, missing_boundaries_input.avif) are used as input.")

    # -------------------------------------------------------------- Q1A
    p.h1("Q1A - The Ghost Signal: Frequency-Domain Image Recovery System")

    p.h2("1. The two-dimensional Discrete Fourier Transform")
    p.body(
        "An M x N grayscale image I(x,y) is a finite 2-D discrete signal whose "
        "intensity varies along both spatial axes. Its 2-D Discrete Fourier "
        "Transform (DFT) and the inverse transform are defined as:")
    p.eq("F(u,v) = Sum_{x=0}^{M-1} Sum_{y=0}^{N-1} I(x,y) * exp[ -j2pi ( ux/M + vy/N ) ]")
    p.eq("I(x,y) = (1/MN) Sum_{u=0}^{M-1} Sum_{v=0}^{N-1} F(u,v) * exp[ +j2pi ( ux/M + vy/N ) ]")
    p.body(
        "The transform is separable: it equals a 1-D DFT applied along the rows "
        "followed by a 1-D DFT along the columns, which is why the FFT can compute "
        "it efficiently. F(u,v) is complex; |F(u,v)| is the magnitude (how much of "
        "spatial frequency (u,v) is present) and its argument is the phase (where "
        "that sinusoid sits). The frequency (u,v) corresponds to a 2-D sinusoidal "
        "'plane wave' across the image, so any image can be written as a weighted "
        "sum of such waves. Low (u,v) near the origin carry slow, large-scale "
        "intensity variation (the bulk of natural-image energy); high (u,v) carry "
        "fine detail, sharp edges and noise.")

    p.h2("2. Why the spectrum must be centered, and how (fftshift)")
    p.body(
        "The raw DFT places the zero-frequency (DC) coefficient F(0,0) - which "
        "equals the sum of all pixel intensities - at the array corner (0,0), and "
        "because the DFT is periodic with period (M,N) the highest frequencies wrap "
        "around to the array edges. This makes the spectrum hard to read. Using the "
        "periodicity/shift property, multiplying I(x,y) by (-1)^(x+y) before the "
        "transform (equivalently, the fftshift operation) moves F(0,0) to the array "
        "centre (M/2, N/2). After centering, frequency increases radially outward "
        "from the middle, so the low-frequency image energy forms a bright central "
        "cluster and periodic-interference components appear as isolated bright dots "
        "at a radius set by their spatial period. This geometric regularity is what "
        "makes the corruption identifiable. We invert with ifftshift before ifft2.")

    p.h2("3. The magnitude spectrum: linear vs decibel scale")
    p.figure("q1a_spectrum.png",
             "Fig 1. Corrupted input (left) and its centered magnitude spectrum on a "
             "linear scale (middle) and a dB scale 20*log10(|F|+eps) (right).")
    p.body(
        "The corrupted image (264 x 517 px) shows a faint message buried under a "
        "dense, regular grid pattern. On the LINEAR magnitude plot essentially "
        "nothing is visible except a single bright point at the centre: the DC term "
        "is orders of magnitude larger than every other coefficient, so it "
        "dominates the colour scale and crushes all the informative structure to "
        "black. Applying the decibel transform 20*log10(|F|+eps) (eps avoids "
        "log(0)) logarithmically compresses the dynamic range. On the dB plot the "
        "answer to 'where are the dominant spectral components located?' becomes "
        "clear: a bright low-frequency blob sits at the centre (the true image "
        "content) and, crucially, a set of sharp, isolated bright dots appears on a "
        "regular lattice away from the centre. The dB scale is therefore not "
        "cosmetic - it is necessary, because the periodic-noise spikes that we must "
        "find are invisible on any linear scale.")

    p.h2("4. How periodic noise manifests, and locating it programmatically")
    p.body(
        "A periodic spatial pattern is, by Fourier's theorem, a sum of 2-D "
        "sinusoids. By the modulation/shift property, an added cosine "
        "cos(2pi(u0 x/M + v0 y/N)) transforms into a pair of Dirac-like impulses at "
        "the symmetric frequencies (+/-u0, +/-v0). Thus periodic corruption that is "
        "spatially spread over the whole image is spectrally CONCENTRATED into a few "
        "sharp, symmetric impulse pairs sitting on a lattice, well away from the "
        "central low-frequency cluster. This is exactly the opposite of genuine "
        "image content, which produces a smooth, continuous energy fall-off "
        "concentrated near DC. That difference - sharp isolated outliers vs. smooth "
        "decay - is the signature we exploit to separate noise from signal.")
    p.figure("q1a_peaks.png",
             "Fig 2. Interference impulses detected programmatically (cyan circles) "
             "on the dB spectrum - a regular lattice away from the DC cluster.")
    p.body(
        "Rather than reading coordinates off the plot by eye, the impulses are "
        "located automatically: (i) the dB spectrum is smoothed with a Gaussian to "
        "estimate the slowly-varying background energy; (ii) this background is "
        "subtracted, leaving only sharp outliers; (iii) points that are both a local "
        "maximum (in a 7x7 window) and lie more than 6 standard deviations above the "
        "residual mean are kept; (iv) a disk of radius 15 px around DC is excluded so "
        "genuine low-frequency content is protected. This returns the impulses, which "
        "indeed lie on a regular grid - the fingerprint of an additive periodic "
        "pattern. Their distance from the centre corresponds to the fundamental "
        "spatial frequency of the corrupting grid.")

    p.h2("5. Designing the frequency-domain rejection system (notch filters)")
    p.body(
        "To suppress the unwanted frequencies while preserving the useful "
        "information we multiply the centered spectrum by a notch-reject mask H(u,v) "
        "that is ~0 in a small neighbourhood around each detected impulse and ~1 "
        "everywhere else. Three profiles trade selectivity against artifacts:")
    p.body(
        "(a) IDEAL (hard cutoff): H = 0 for distance D < D0, else 1. It removes the "
        "impulse completely but the abrupt cutoff is equivalent to multiplying by a "
        "rectangular window, whose spatial-domain counterpart is a sinc; this "
        "produces Gibbs RINGING (oscillating halos) around sharp edges in the "
        "recovered image.")
    p.eq("Gaussian:  H(u,v) = Product over impulses [ 1 - exp( -D^2 / (2 D0^2) ) ]")
    p.body(
        "(b) GAUSSIAN: a smooth notch with a gradual roll-off. Because its transform "
        "is again a smooth Gaussian (no side-lobes), it suppresses the impulse with "
        "minimal ringing. (c) BUTTERWORTH of order n: H = 1/(1+(D0/D)^(2n)), whose "
        "sharpness is tunable by n - it interpolates between the Gaussian and ideal "
        "behaviours. We justify the chosen notch radius (D0 = 6 px) as wide enough to "
        "cover each impulse plus its immediate skirt, yet narrow enough not to bite "
        "into nearby legitimate frequencies; Section 8 verifies this choice "
        "quantitatively.")

    p.h2("6. Reconstruction and the required comparison")
    p.body(
        "After masking we invert: ifftshift to undo the centering, then ifft2, and "
        "we take the real part (the tiny imaginary residue is numerical noise). The "
        "four-panel comparison required by the brief is shown below.")
    p.figure("q1a_grid.png",
             "Fig 3. Required grid: corrupted image | spectrum (dB) | filtered "
             "spectrum (dB) | recovered image.")
    p.body(
        "The result answers 'has the hidden message been recovered?' decisively: "
        "the recovered panel cleanly reveals the previously-illegible text -")
    p.set_font("DejaVu", "B", 12); p._mc(7, '   "QUIZ 2 ON 7th JULY IN TUTORIAL HOURS"')
    p.body(
        "In the filtered spectrum the lattice of bright impulses has been removed "
        "while the central image energy is untouched, which is precisely why the "
        "grid disappears spatially but the scene content survives.")

    p.h2("7. Ideal vs. Gaussian notch: the ringing trade-off")
    p.figure("q1a_ringing.png",
             "Fig 4. Reconstruction with an ideal (hard) notch vs. a Gaussian notch. "
             "The ideal cutoff introduces faint ringing; the Gaussian is cleaner.")
    p.body(
        "Comparing the two confirms the theory of Section 5: the ideal notch leaves "
        "subtle ringing/halo artifacts (the Gibbs phenomenon) near high-contrast "
        "strokes, whereas the Gaussian notch yields a visibly smoother background. "
        "For this task the Gaussian (or a low-order Butterworth) is the better "
        "engineering choice.")

    p.h2("8. Filter-parameter study: does removing more frequencies always help?")
    p.figure("q1a_radius_sweep.png",
             "Fig 5. Recovered image as the notch radius grows, with the percentage "
             "of spectral energy removed (bottom-right curve).")
    p.body(
        "Sweeping the notch radius answers the brief's key question directly. For "
        "very small radii (< ~4 px) the impulses are only partially attenuated and "
        "faint residual grid lines survive in the output. In an intermediate band "
        "(~4-8 px) the message is crisp and the background clean. As the radius "
        "grows large (>~16 px) the notches begin to overlap the central "
        "low-frequency cluster and start deleting GENUINE image frequencies; the "
        "energy-removed curve climbs and the recovered text visibly softens and "
        "blurs. Conclusion: removing more frequencies does NOT always improve the "
        "result - there is an optimum. Too little filtering leaves noise; too much "
        "destroys useful information. This is the central practical lesson of "
        "frequency-domain restoration and the justification for the moderate radius "
        "used in Fig 3.")

    p.h2("9. Practical applications")
    p.body(
        "Frequency-domain restoration of this kind is used wherever periodic or "
        "narrowband interference corrupts a signal: removing carrier/tone "
        "interference and moire in COMMUNICATION SYSTEMS; de-striping and "
        "sensor-pattern removal in REMOTE SENSING and satellite imagery; cleaning "
        "periodic structured noise in SURVEILLANCE footage; and suppressing "
        "RF-interference 'zipper'/spike artifacts and periodic ghosting in MRI and "
        "other MEDICAL IMAGING. In each case the same principle holds - periodic "
        "corruption is diffuse in space but concentrated in frequency, so it is far "
        "easier to excise in the Fourier domain than in the spatial domain.")

    # -------------------------------------------------------------- Q1B
    p.add_page()
    p.h1("Q1B - Missing Boundaries: Sobel Edge Detection")

    p.h2("1. Edges, gradients, and the operation that detects them")
    p.body(
        "An edge is a location where image intensity changes abruptly; such "
        "transitions usually coincide with object boundaries and carry a large share "
        "of the structural information in a scene (which is why humans, and systems "
        "for autonomous driving, medical imaging, face recognition and industrial "
        "inspection, rely on them). The mathematical operation that identifies rapid "
        "intensity change is DIFFERENTIATION: the spatial gradient")
    p.eq("grad I = ( dI/dx , dI/dy ),   |grad I| = sqrt( (dI/dx)^2 + (dI/dy)^2 )")
    p.body(
        "points in the direction of steepest intensity increase and its magnitude is "
        "large exactly where intensity changes fast. So edge detection reduces to "
        "estimating the two partial derivatives of the discrete image and combining "
        "them.")

    p.h2("2. The Sobel kernels and 2-D convolution")
    p.body(
        "The Sobel operator estimates the partials with two 3x3 kernels that combine "
        "a central difference (the derivative) along one axis with a [1 2 1] "
        "smoothing along the perpendicular axis (which suppresses noise):")
    p.eq("Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]    Gy = Gx^T = [[-1,-2,-1],[0,0,0],[1,2,1]]")
    p.body(
        "Gx responds to vertical edges (horizontal intensity change), Gy to "
        "horizontal edges. Each kernel is separable, Gx = [1,2,1]^T * [-1,0,1], which "
        "is why it is both a smoother and a differentiator. The responses are "
        "obtained by 2-D convolution of the kernel with the image,")
    p.eq("gx = Gx * I,   gy = Gy * I,   magnitude = sqrt(gx^2 + gy^2),   "
         "theta = atan2(gy, gx)")
    p.body(
        "where theta gives the local edge orientation. The convolution is "
        "implemented from scratch (a reflect-padded sliding-window tensor "
        "contraction) rather than calling a library edge routine; reflect "
        "('symmetric') padding handles the borders without injecting spurious edges. "
        "To validate the implementation we cross-check it against SciPy's "
        "convolve2d: the maximum absolute difference of the two Sobel magnitudes is "
        "0.0, confirming the manual convolution is exactly correct.")

    p.h2("3. Applying Sobel to the image")
    p.figure("q1b_sobel.png",
             "Fig 6. Input; horizontal response gx; vertical response gy; gradient "
             "magnitude; gradient direction (rad); and the thresholded edge map.")
    p.body(
        "Applied to the provided image, gx highlights vertical building/structure "
        "edges and gy the horizontal ones; their combination yields the gradient "
        "magnitude, in which the boundaries of the building, the sculpture and the "
        "window frames stand out clearly against flat regions. Normalising the "
        "magnitude to [0,255] and applying a threshold (T = 60) isolates the strong "
        "boundaries into a clean binary edge map - revealing the 'missing "
        "boundaries' the brief asks for. The direction map encodes orientation and "
        "is the quantity a follow-on method (e.g. non-maximum suppression) would use "
        "to thin edges.")

    p.h2("4. How noise affects edge detection - and why")
    p.figure("q1b_noise.png",
             "Fig 7. Sobel applied directly (no smoothing) to the image with "
             "increasing additive Gaussian noise (sigma = 0, 15, 35).")
    p.body(
        "Differentiation is a HIGH-PASS operation: in the frequency domain a "
        "derivative multiplies the spectrum by a factor proportional to frequency "
        "magnitude, so it AMPLIFIES high frequencies. Noise is broadband and "
        "high-frequency-heavy, so running Sobel directly on a noisy image amplifies "
        "the noise into a storm of spurious, speckle-like edges - visible in Fig 7, "
        "where the edge map degrades rapidly as the noise standard deviation grows. "
        "Raw gradient edge detection is therefore fragile to noise.")

    p.h2("5. The role of pre-smoothing, and the noise-vs-detail trade-off")
    p.figure("q1b_smoothing.png",
             "Fig 8. A fixed noisy input (sigma = 35) pre-smoothed with a Gaussian "
             "of increasing sigma before Sobel (sigma = 0, 1, 2, 3.5).")
    p.body(
        "The standard remedy is to LOW-PASS filter before differentiating. A "
        "Gaussian blur is a low-pass filter - its Fourier transform is again a "
        "Gaussian that decays with frequency - so it removes the high-frequency "
        "noise that the derivative would otherwise amplify. Fig 8 shows the effect "
        "and the inherent compromise: with no smoothing the edge map is dominated by "
        "speckle; light smoothing (sigma ~= 1-2) removes most false edges while "
        "keeping the strong true boundaries; heavy smoothing (sigma = 3.5) gives the "
        "cleanest noise rejection but also blurs away fine and weak edges and thins "
        "the contours. This is the NOISE-vs-DETAIL trade-off: smoothing buys noise "
        "immunity at the cost of localisation and weak-edge sensitivity.")

    p.h2("6. Can weak edges be preserved without excessive noise?")
    p.figure("q1b_canny_bilateral.png",
             "Fig 9. Input; Sobel with Gaussian pre-smoothing; Canny (non-maximum "
             "suppression + hysteresis); Sobel with an edge-preserving bilateral "
             "pre-filter.")
    p.body(
        "Partly - and a single global magnitude threshold forces one compromise "
        "everywhere, so it cannot. Two improvements answer the question concretely. "
        "(i) The CANNY detector adds non-maximum suppression (which thins each ridge "
        "to a one-pixel line by keeping only local maxima along the gradient "
        "direction) and HYSTERESIS thresholding (two thresholds: strong edges are "
        "always kept, weak edges are kept only if connected to a strong one). This "
        "preserves faint-but-real edges while rejecting isolated noise, which a "
        "single Sobel threshold cannot. (ii) A BILATERAL filter is an "
        "edge-preserving alternative to Gaussian pre-smoothing: it averages only "
        "over nearby pixels of SIMILAR intensity, so it smooths flat regions while "
        "refusing to blur across strong edges, giving crisper boundaries. Fig 9 "
        "shows Canny producing clean, thin, well-connected contours that recover "
        "weak edges the plain Sobel threshold misses, and the bilateral pre-filter "
        "yielding sharper Sobel edges than Gaussian smoothing.")

    p.h2("7. Conclusions (Q1)")
    p.body(
        "Q1A: periodic corruption is a small set of symmetric impulse pairs on a "
        "lattice in the centered, dB-scaled spectrum, distinct from the smooth "
        "near-DC image energy; notch-rejecting them and inverting the DFT recovers "
        "the hidden message, with a Gaussian notch of moderate radius giving the "
        "best quality (too large destroys content). Q1B: edges are high-gradient "
        "regions estimated with the Sobel kernels via 2-D convolution; the derivative "
        "is high-pass and amplifies noise, so a low-pass (Gaussian or edge-preserving "
        "bilateral) pre-filter is needed, trading detail for noise immunity, with "
        "Canny's NMS + hysteresis the principled way to keep weak edges.")
    p.output(os.path.join(REPORT, "Q1_report.pdf"))
    print("Wrote Q1_report.pdf")


# ===========================================================================
# Q2 REPORT
# ===========================================================================
def build_q2():
    p = Report("Q2 - The Midnight Episode: ECG Arrhythmia Detection")
    title_block(p, "'Catching the Arrhythmia' via normalized template correlation (7.5%) "
                   "- EE200 Course Project")
    p.body(
        "A Holter recording is the discrete-time signal x[n] sampled at fs = 250 Hz "
        "with N = 5000 samples; in the healthy stretch one full beat {P, QRS, T} "
        "arrives every 0.8 s. The strategy: learn one clean beat as a template, slide "
        "it along the recording, and flag the moment the shape stops matching. The "
        "analysis uses the provided files patient_ecg.npy and template.npy "
        "exclusively; every numeric value below is computed in the notebook, not "
        "estimated. Detector functions live in src/q2_ecg.py.")
    p.figure("q2_ecg_full.png",
             "Fig 1. The full 20 s recording: regular, identical beats give way to an "
             "irregular region partway through.")

    p.h2("(a) Reading the signal [0.75%]")
    p.body(
        "(i) Duration. The clip length is N/fs = 5000/250 = 20.0 seconds.\n"
        "(ii) Heart rate and beat length. One beat every 0.8 s is a rate of "
        "60/0.8 = 75 beats per minute, and one beat occupies 0.8 s * 250 Hz = 200 "
        "samples.\n"
        "(iii) Fundamental frequency. Treating the healthy ECG as periodic with "
        "period T0 = 0.8 s, the fundamental frequency is f0 = 1/T0 = 1/0.8 = 1.25 Hz. "
        "(Consistently, the 200-sample template is exactly one period long.)")

    p.h2("(b) The healthy heart in the frequency domain [0.75%]")
    p.figure("q2_spectrum.png",
             "Fig 2. Magnitude spectrum of the healthy segment; red lines mark "
             "multiples of f0 = 1.25 Hz.")
    p.body(
        "(i) Shape of |X(f)|. A periodic (or nearly periodic) signal does NOT have a "
        "smooth continuous spectrum: by Fourier-series theory its energy is "
        "concentrated at discrete HARMONICS of the fundamental, f0, 2f0, 3f0, .... "
        "Fig 2 confirms a line/comb spectrum with peaks landing on the red multiples "
        "of 1.25 Hz, not a smooth curve.\n"
        "(ii) What drives the high frequencies. The QRS complex is a sharp, narrow "
        "spike, whereas the P and T waves are broad and smooth. A narrow feature in "
        "time requires a broad band of frequencies to synthesise (the "
        "time-bandwidth / uncertainty principle), so the QRS is responsible for the "
        "high-frequency content; the broad P and T waves are inherently "
        "low-frequency.\n"
        "(iii) Raising the rate to 150 bpm. The period halves to T0 = 0.4 s, so "
        "f0 = 1/0.4 = 2.5 Hz: the fundamental DOUBLES, and since the harmonic "
        "spacing equals f0, the spacing between spectral lines doubles as well (the "
        "comb stretches out).")

    p.h2("(c) Cutting a heartbeat - windowing [1.0%]")
    p.figure("q2_windows.png",
             "Fig 3. The 200-sample template (one beat) and the two mis-sized "
             "rectangular windows on the recording.")
    p.body(
        "Maya isolates a template by multiplying the signal with a rectangular "
        "window (1 inside [n1,n2], 0 outside).\n"
        "(i) Correct width and placement. To capture exactly one full beat the "
        "window should be about one beat period wide, ~200 samples, positioned to "
        "bracket a complete {P, QRS, T} cycle starting just before the P wave.\n"
        "(ii) What goes wrong at 80 and 600 samples. An 80-sample window is far too "
        "SHORT: it slices into the beat itself, clipping part of the QRS/T so the "
        "template is incomplete and distorted, and its hard rectangular edges cut "
        "through signal, causing spectral leakage. A 600-sample window is too LONG: "
        "at 200 samples/beat it spans roughly three beats, so it swallows fragments "
        "of neighbouring beats and contaminates the template with adjacent-beat "
        "energy.\n"
        "(iii) Why shortest is not best. This is the same time-frequency uncertainty "
        "trade-off as the STFT: a short window gives sharp time localisation but poor "
        "frequency resolution, a long window the reverse. Here making the window 'as "
        "short as possible' is actively harmful because it literally cuts through the "
        "beat we are trying to model - so the right width is matched to the beat, not "
        "minimised.")

    p.h2("(d) Matching the template - normalized correlation [1.5%]")
    p.eq("rho(m) = [ Sum_k t[k] x[m+k] ] / ( ||t|| * ||x_m|| )")
    p.body(
        "rho(m) is the cosine similarity between the template t and the length-L "
        "segment of x starting at m; ||.|| is the root-sum-of-squares energy.\n"
        "(i) Range and perfect-match value. By the Cauchy-Schwarz inequality "
        "rho(m) lies in [-1, +1]; rho = +1 signals a near-perfect shape match "
        "(template and segment are scalar multiples of each other).\n"
        "(ii) Why normalize. Real ECG amplitude drifts slowly (baseline wander) and "
        "beats vary in size, so we want a score that measures SHAPE, not amplitude. "
        "Dividing by the two energies makes rho invariant to scaling: a perfectly "
        "shaped beat that happens to be twice as tall still scores rho = +1. The "
        "UNNORMALIZED dot product, by contrast, would roughly DOUBLE for that same "
        "beat - it conflates 'right shape' with 'large amplitude', which baseline "
        "wander then makes unreliable. (Verified numerically: rho(t, 2t) = +1 while "
        "the raw dot product scales by 2x.)\n"
        "(iii) An inverted beat. If an abnormal beat is flipped upside-down relative "
        "to the template, t and the segment are anti-aligned, so rho ~= -1. Because "
        "that is at the opposite extreme from the healthy rho ~= +1, inverted beats "
        "are especially easy to flag with a simple threshold.")

    p.h2("(e) Onset detection and the spectrogram [1.5%]")
    p.figure("q2_correlation.png",
             "Fig 4. ECG with the detected onset (top) and rho vs. time (bottom): the "
             "first 12 beats score ~= +1, then the score drops to ~= -1.")
    p.body(
        "(i) An onset rule and its threshold trade-off. Declare the arrhythmia onset "
        "at the first beat whose correlation falls below a threshold (0.5, as fixed "
        "in part g). Setting the threshold too HIGH causes false alarms from normal "
        "beat-to-beat variability and noise; setting it too LOW makes the detector "
        "under-sensitive and it misses mild but real morphology changes. 0.5 sits "
        "safely between the healthy cluster (rho ~= +1) and the abnormal beats.\n"
        "(ii) The spectrogram view. In the healthy region the spectrogram shows "
        "clean, steady horizontal harmonic bands at f0, 2f0, 3f0, ...; in the "
        "arrhythmia region those bands smear, break up, shift and are joined by "
        "irregular new energy as the periodicity is lost.\n"
        "(iii) Why they can disagree, and which to trust. The correlation score is "
        "evaluated per beat, so it can pinpoint the exact beat where the shape "
        "breaks. The spectrogram must accumulate several irregular beats inside its "
        "analysis window before the harmonic structure visibly degrades, so its "
        "estimate of the onset is inherently smeared and delayed. To pinpoint the "
        "single moment a bad beat first appears, trust the CORRELATION; use the "
        "spectrogram as the corroborating big-picture view. (This is again the "
        "windowing time-resolution trade-off from part (c).)")

    p.h2("(f) Sampling and aliasing [0.5%]")
    p.body(
        "The clinically important QRS features contain content up to about 40 Hz.\n"
        "(i) Nyquist minimum. To capture content up to 40 Hz without aliasing the "
        "sampling rate must be at least 2 * 40 = 80 Hz.\n"
        "(ii) What 50 Hz does. 50 Hz is below 80 Hz, so any QRS content above the "
        "new Nyquist frequency of 25 Hz folds back ('aliases') into lower "
        "frequencies, distorting and smearing the sharp spike or injecting spurious "
        "low-frequency components. This is dangerous because the detector relies on "
        "the SHAPE of the beat through correlation, and aliasing corrupts exactly "
        "that shape - causing false positives and false negatives.\n"
        "(iii) The fix and its cost. If the rate must be lowered, apply an "
        "anti-aliasing low-pass filter (cutoff ~= 25 Hz, below the new Nyquist) "
        "BEFORE downsampling. The unavoidable cost is that genuine fine QRS detail "
        "above 25 Hz is filtered out - real diagnostic morphology is permanently "
        "lost in exchange for the lower data rate.")

    p.h2("(g) Prototyping the detector: find_onset [1.5%]")
    p.body(
        "find_onset(ecg_signal, template, threshold) implements the rule directly: "
        "it steps through the recording BEAT BY BEAT (advancing by len(template) = "
        "200 samples each time, not sample-by-sample), computes the normalized "
        "correlation of each beat against the template, and returns the start index "
        "of the first beat whose score drops strictly below the threshold; if no beat "
        "breaches it, it returns -1. On the provided data with threshold = 0.5 it "
        "returns sample index 2400, i.e. t = 2400/250 = 9.6 s (the 13th beat). Read "
        "from Fig 4, the first 12 beats score rho ~= +0.99 and the beat at 9.6 s "
        "scores rho ~= -0.99 - an INVERTED beat - after which the recording stays "
        "irregular. The arrhythmia therefore begins at t = 9.6 s. Sanity checks: a "
        "threshold above every score returns the first beat, and an impossible "
        "threshold returns -1.")

    p.h2("(h) Visualising the spectrogram, and choosing nperseg [0.5%]")
    p.figure("q2_spectrogram.png",
             "Fig 5. Spectrogram (nperseg = 500): steady harmonic bands in the "
             "healthy region break up at the arrhythmia onset (marked).")
    p.body(
        "The spectrogram is computed with scipy.signal.spectrogram and plotted with "
        "matplotlib. The critical parameter is the window length nperseg, which sets "
        "the frequency resolution df = fs/nperseg. To resolve the harmonic spacing "
        "f0 = 1.25 Hz we need df <~ 1.25 Hz, i.e. nperseg >~ fs/f0 = 250/1.25 = 200. "
        "We choose nperseg = 500, giving df = 250/500 = 0.5 Hz - comfortably finer "
        "than 1.25 Hz - so the harmonics appear as clean, well-separated horizontal "
        "bands in the healthy region, exactly as needed to see them break up at the "
        "onset. (A much smaller window would blur the harmonics together; a much "
        "larger one would smear the onset in time.)")

    p.h2("Conclusion (Q2)")
    p.body(
        "On the real recording: 20 s clip, 75 bpm, 200 samples/beat, f0 = 1.25 Hz. "
        "The normalized correlation stays at ~= +1 for the first twelve beats and "
        "then the arrhythmia begins at t = 9.6 s (sample 2400) with an inverted beat "
        "(rho ~= -1); the spectrogram independently shows the harmonic bands "
        "disintegrating at that point. The per-beat correlation pinpoints the exact "
        "onset, the spectrogram corroborates it.")
    p.output(os.path.join(REPORT, "Q2_report.pdf"))
    print("Wrote Q2_report.pdf")


# ===========================================================================
# Q3 REPORT
# ===========================================================================
def build_q3():
    p = Report("Q3 - Sonic Signatures: Audio Fingerprinting + Identifier App")
    title_block(p, "Q3A 'Magical Mystery Tune' (7.5%)  +  Q3B 'Zapptain America' (5%) "
                   "- EE200 Course Project")
    p.body(
        "This builds a Shazam-style identifier that never compares raw waveforms: it "
        "converts audio to a time-frequency picture, keeps only a sparse set of "
        "standout peaks as a fingerprint, and matches fingerprints by time "
        "alignment. The real provided library of 50 songs is indexed from "
        "data/songs/ (filenames kept exactly as given - the filename without "
        "extension is the ground-truth label); the full library produces about 2.47 "
        "million paired hashes. The pipeline lives in src/q3_fingerprint.py; "
        "parameters quoted below (sample rate 11025 Hz, STFT window 1024 with 50% "
        "overlap, peak neighbourhood 20 bins, fan-out 15, target-zone gap 1-40 "
        "frames) are defined there with justification.")

    p.h1("Q3A - Sonic Signatures")

    p.h2("1. Why a single Fourier transform is not enough")
    p.figure("q3_dft.png",
             "Fig 1. DFT magnitude of an entire song: it reveals WHICH frequencies "
             "are present, but not WHEN.")
    p.body(
        "The DFT of a whole song tells you which frequencies the song contains but "
        "completely discards timing - the same set of notes in any order gives a "
        "similar magnitude spectrum. Since recognition depends on the pattern of "
        "frequencies OVER TIME, a single DFT is insufficient. We need to watch the "
        "frequency content evolve as the song plays.")

    p.h2("2. The spectrogram and the time-frequency resolution trade-off")
    p.figure("q3_window_tradeoff.png",
             "Fig 2. Spectrogram of the song with a SHORT window (left) and a LONG "
             "window (right).")
    p.body(
        "A spectrogram restores timing: slide a short window along the signal and "
        "take the DFT of each windowed slice, then stack the slices side by side so "
        "time runs horizontally, frequency vertically, and brightness shows how "
        "strong each frequency is at each instant (a rising note traces a rising "
        "streak, a steady tone a horizontal line). Each column is just an ordinary "
        "DFT of one short piece. Recomputing with a short vs. a long window exposes "
        "the resolution trade-off: a SHORT window localises events sharply in time "
        "but smears them in frequency (wide, blurry bands); a LONG window resolves "
        "frequency finely but smears events in time. One cannot have both at once - "
        "the same uncertainty principle seen in Q2(c). The fingerprinter picks an "
        "intermediate window (1024 samples at 11025 Hz, ~93 ms with 50% overlap) to "
        "balance the two.")

    p.h2("3. The fingerprint: a constellation of spectral peaks")
    p.figure("q3_constellation.png",
             "Fig 3. The constellation - the strongest local-maximum peaks - overlaid "
             "on the spectrogram.")
    p.body(
        "From the spectrogram we keep only the strongest PEAKS: points that are a "
        "local maximum within a neighbourhood (a 20-bin maximum filter is compared "
        "against the spectrogram) and that exceed an amplitude floor. These sparse, "
        "high-energy points - the 'constellation' - are what survive distortion: "
        "even when noise or compression fills in the quiet regions, the loudest "
        "peaks remain in place. Discarding everything else also makes the "
        "representation compact.")

    p.h2("4. Hashing: pairing peaks into compact, specific keys")
    p.body(
        "A lone peak frequency is not distinctive - the same note recurs in countless "
        "songs. So each ANCHOR peak is paired with several nearby peaks lying ahead "
        "of it in a time-frequency 'target zone' (here, a forward time gap of 1-40 "
        "spectrogram frames, up to 15 partners per anchor). Each pair becomes a "
        "compact hash key (f1, f2, dt) - two frequencies and the time gap between "
        "them - with value (song, t_anchor). Building this for every song yields one "
        "inverted-index database mapping each hash to the list of (song, time) where "
        "it occurs. A joint triple (f1, f2, dt) is far rarer than a single frequency, "
        "so it is far more SPECIFIC.")

    p.h2("5. Matching by time-offset alignment")
    p.figure("q3_offset_hist.png",
             "Fig 4. Offset histograms: the correct song's matched hashes pile up at "
             "ONE time offset (a sharp spike); a wrong song gives only scattered "
             "counts.")
    p.body(
        "A query clip is fingerprinted the same way. For every query hash that hits "
        "the database, we record offset = t_database - t_query for the matching song "
        "and accumulate these offsets into a per-song histogram. The reasoning: if "
        "the query truly comes from a song, ALL of its matching hashes occur at the "
        "same fixed lag into that song, so their offsets coincide and the histogram "
        "shows one tall spike. For a wrong song, the few coincidental hash matches "
        "occur at random lags, so the histogram is a low, flat noise floor. The "
        "predicted song is the one with the tallest spike, and the spike height "
        "(divided by the runner-up's) gives a confidence. On the real library a "
        "4-second query taken from 'A Hard Day_s Night' is identified correctly with "
        "a confidence of ~34x the runner-up; every built-in sample clip is "
        "identified correctly.")

    p.h2("6. Single peaks vs. paired hashes - why pairing wins")
    p.body(
        "Repeating the match using SINGLE peak frequencies as the key (no pairing) "
        "produces a much noisier, flatter offset histogram even for the correct "
        "song. The reason is specificity: an individual frequency bin recurs "
        "constantly across unrelated songs, so single-peak matches collide by chance "
        "very often and the true alignment is buried in noise. Pairing constrains "
        "(f1, f2, dt) jointly, a coincidence that is far rarer, so the correct "
        "alignment dominates. Quantitatively, on the same clip the paired matcher is "
        "roughly an order of magnitude more decisive than the single-peak matcher "
        "(e.g. ~500x vs. ~30x winner-to-runner-up ratio in the app's live "
        "comparison) - which is exactly why joining two peaks into one fingerprint "
        "makes a correct match so much more confident.")

    p.h2("7. Robustness experiments")
    p.figure("q3_noise_robustness.png",
             "Fig 5. Correct-song match score as additive white noise increases "
             "(SNR decreasing left to right).")
    p.body(
        "ADDITIVE NOISE. Adding white noise at decreasing SNR and re-identifying, the "
        "match score degrades GRACEFULLY and recognition survives surprisingly heavy "
        "noise before collapsing past a breakdown SNR. This robustness is a direct "
        "consequence of the peak constellation: the loudest spectral peaks remain the "
        "loudest until the noise is strong enough to bury them, so the fingerprint is "
        "largely intact at moderate noise.")
    p.body(
        "PITCH SHIFT / TIME STRETCH. Shifting the query up in pitch by even a small "
        "amount, by contrast, DEFEATS the identifier almost immediately (on the real "
        "library a +0.5-semitone shift already causes a wrong prediction). The reason "
        "is fundamental: the hash keys store ABSOLUTE frequency bins, and a pitch "
        "shift multiplies every frequency by a constant, moving every peak into a "
        "different bin - so essentially no hash collides with the database any more. "
        "A human, who perceives RELATIVE pitch relationships, hears the same song, "
        "but the absolute-frequency fingerprint no longer matches. (A pure time "
        "stretch similarly disturbs the dt component of the hashes.)")
    p.body(
        "A FIX. Quantise peaks onto a LOG-frequency / constant-Q axis: there a pitch "
        "shift becomes a constant ADDITIVE offset that can be tolerated or searched "
        "over, instead of a multiplicative scaling that scrambles every bin. "
        "Alternatives are to allow a small +/- tolerance in the hash lookup, or to "
        "pre-index a few pitch-shifted copies of each song offline.")

    p.add_page()
    p.h1("Q3B - 'Zapptain America': the identifier app")
    p.body(
        "The identifier is wrapped in a deployed, interactive web application "
        "(custom dark-themed Flask backend + HTML/CSS/JS frontend) that reuses "
        "src/q3_fingerprint.py with no duplicated logic. It indexes the song library "
        "once into the shipped hash database so it works immediately on load.")
    p.h2("Required modes")
    p.body(
        "SINGLE-CLIP MODE (Identify tab): the user uploads a query clip (or records "
        "one live from the microphone, or picks a built-in sample) and the app "
        "displays, in order, the intermediate steps the brief asks for - the "
        "spectrogram, the constellation of peaks, and the offset histogram that "
        "decides the match - followed by the predicted song name and a confidence "
        "score, plus a per-stage timing breakdown and the ranked candidate songs.\n"
        "BATCH MODE (Batch tab): the user uploads a set of query clips; each is "
        "identified against the indexed library and the results are written to a "
        "standardised results.csv with EXACTLY two columns, filename and prediction, "
        "where prediction is the matched track's filename without extension (or "
        "'none' when no candidate clears the confidence threshold). This exact "
        "format was verified for the automatic grader.\n"
        "LIBRARY MODE: shows the whole indexed library as cards (each with its "
        "constellation thumbnail and hash count).")
    p.h2("Extra analysis exposed in the app")
    p.body(
        "Beyond the brief, the app makes the Q3A analysis interactive: a Robustness "
        "Lab runs the noise-SNR and pitch-shift sweeps on the user's own clip and "
        "charts how the score holds up, and a single-peak-vs-paired-hash panel shows "
        "the two offset histograms side by side on the live query so the "
        "specificity argument of Section 6 can be seen directly.")
    p.h2("Deployment")
    p.body(
        "The app is deployed on Hugging Face Spaces (Docker), which provides ample "
        "memory for the librosa/numba stack and avoids cold-start; the indexed "
        "database ships with the app so it is usable immediately. Both modes were "
        "tested end-to-end on the live link.")
    p.body("")
    p.set_font("DejaVu", "B", 11)
    p._mc(6, "LIVE APP:  https://mayur2403-ee200-audio-fingerprinting.hf.space")
    p._mc(6, "SOURCE CODE:  https://github.com/Mayurdp2403/ee200-course-project")
    p.output(os.path.join(REPORT, "Q3_report.pdf"))
    print("Wrote Q3_report.pdf")


if __name__ == "__main__":
    build_q1(); build_q2(); build_q3()
    print("All reports written to", REPORT)
    if _LEFTOVER:
        print("WARNING unmapped codepoints:", sorted(hex(ord(c)) for c in _LEFTOVER))
