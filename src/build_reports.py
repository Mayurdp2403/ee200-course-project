"""Generate the per-question PDF reports in report/ using fpdf2 (pure-Python).

Reports embed the figures produced by the executed notebooks and answer every
sub-question explicitly, in order. Run the notebooks first so figures/ is fresh.
"""
import os

import matplotlib
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
REPORT = os.path.join(ROOT, "report")
os.makedirs(REPORT, exist_ok=True)

# Unicode-capable fonts shipped with matplotlib (no system install needed).
TTF_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_REG = os.path.join(TTF_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(TTF_DIR, "DejaVuSans-Bold.ttf")

# This fpdf2/font build only renders Latin-1 reliably; transliterate the math
# symbols used in the prose to clean ASCII so the PDFs build and read correctly.
_SUBS = {
    "ρ": "rho", "σ": "sigma", "√": "sqrt", "∇": "grad", "Σ": "Sum", "Δ": "Delta",
    "ω": "w", "θ": "theta", "π": "pi", "ε": "eps", "μ": "u", "∂": "d", "φ": "phi",
    "⇒": "=>", "→": "->", "≈": "~=", "≥": ">=", "≤": "<=", "×": "x", "−": "-",
    "·": "-", "²": "^2", "³": "^3", "°": "deg", "•": "*", "—": "-", "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "′": "'", "ᵀ": "^T", "‖": "||",
    "∈": " in ", "±": "+/-", "≲": "<~", "≳": ">~", "₀": "0", "…": "...",
    "∝": " proportional to ",
}


_LEFTOVER: set = set()


def _s(txt: str) -> str:
    """Transliterate math symbols to ASCII for the Latin-1 PDF font."""
    for k, v in _SUBS.items():
        txt = txt.replace(k, v)
    # Backstop: replace any still-unmapped non-ASCII char with '?' and record it.
    out = []
    for ch in txt:
        if ord(ch) < 128:
            out.append(ch)
        else:
            _LEFTOVER.add(ch)
            out.append("?")
    return "".join(out)


class Report(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font("DejaVu", "", FONT_REG)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_page()

    def header(self):
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(120)
        self.cell(0, 6, _s(self.title_text), align="R")
        self.ln(8)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(150)
        self.cell(0, 6, f"Mayur - Roll 240643 - EE200 - page {self.page_no()}", align="C")

    def _mc(self, h, txt, **kw):
        """multi_cell with x reset to the left margin and an explicit width."""
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, h, _s(txt), **kw)

    def h1(self, txt):
        self.set_font("DejaVu", "B", 15); self.ln(2)
        self._mc(8, txt); self.ln(1)

    def h2(self, txt):
        self.set_font("DejaVu", "B", 12); self.ln(1)
        self._mc(7, txt); self.ln(0.5)

    def body(self, txt):
        self.set_font("DejaVu", "", 10.5)
        self._mc(5.4, txt); self.ln(1)

    def figure(self, name, caption, w=170):
        path = os.path.join(FIG, name)
        if not os.path.exists(path):
            self.body(f"[figure {name} not found — run the notebook first]"); return
        if self.get_y() + w * 0.45 > 270:
            self.add_page()
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.set_font("DejaVu", "", 9); self.set_text_color(90)
        self._mc(4.5, caption, align="C"); self.set_text_color(0); self.ln(2)


def title_block(pdf, subtitle):
    pdf.set_font("DejaVu", "B", 18)
    pdf._mc(9, pdf.title_text)
    pdf.set_font("DejaVu", "", 11); pdf.set_text_color(80)
    pdf._mc(6, subtitle + "\nMayur - Roll 240643 - EE200 Signals, Systems & Networks - IIT Kanpur")
    pdf.set_text_color(0); pdf.ln(3)


# =========================================================== Q1 report
def build_q1():
    p = Report("Q1 — Image Recovery & Edge Detection")
    title_block(p, "Q1A Frequency Forensics (Ghost Signal) + Q1B Sobel Edge Detection")

    p.h1("Q1A — Frequency-Domain Image Recovery")
    p.h2("Methodology")
    p.body(
        "An image I(x,y) is a finite 2-D discrete signal. Its 2-D DFT is\n"
        "    F(u,v) = ΣΣ I(x,y) · exp(−j2π(ux/M + vy/N)),\n"
        "and the inverse divides by MN and uses +j. The raw DFT puts DC at the "
        "corner, so we apply fftshift to centre it: low frequencies move to the "
        "middle and spatial frequency grows radially outward.\n\n"
        "A periodic spatial pattern added to the image transforms (shift/modulation "
        "property) into a few sharp, symmetric impulse PAIRS away from the centre, "
        "whereas real image content is a smooth energy falloff near DC. So periodic "
        "noise is spatially diffuse but spectrally concentrated — which is exactly "
        "why we remove it in the Fourier domain.")
    p.figure("q1a_spectrum.png",
             "Fig 1. Corrupted input and its magnitude spectrum on linear vs dB scales. "
             "On the linear scale the DC term hides everything; the dB scale "
             "(20·log10|F|) exposes the periodic-noise impulse lattice.")
    p.body(
        "Why dB is necessary: the single DC term dominates the linear magnitude so "
        "the periodic-noise spikes are invisible; the log compression reveals them "
        "as bright dots on a regular lattice away from the centre.")
    p.figure("q1a_peaks.png",
             "Fig 2. Interference impulses located programmatically (subtract a "
             "smoothed background, keep sharp local-maxima outliers above 6σ, exclude "
             "a disk around DC). Not eyeballed.")
    p.h2("Notch filtering & reconstruction")
    p.body(
        "We attenuate a neighbourhood around each impulse with three profiles: "
        "ideal (hard cutoff — sharpest but rings via Gibbs), Gaussian "
        "(H = 1 − exp(−D²/2D0²), smooth, minimal ringing), and Butterworth "
        "(tunable). Inverse = ifftshift → ifft2 → real part.")
    p.figure("q1a_grid.png",
             "Fig 3. Required 4-panel grid: corrupted | spectrum (dB) | filtered "
             "spectrum (dB) | recovered. The hidden message is recovered:")
    p.body("RECOVERED HIDDEN MESSAGE:  \"QUIZ 2 ON 7th JULY IN TUTORIAL HOURS\".")
    p.figure("q1a_ringing.png",
             "Fig 4. Ideal vs Gaussian notch: the ideal hard cutoff introduces "
             "visible ringing; the Gaussian notch is smoother.")
    p.figure("q1a_radius_sweep.png",
             "Fig 5. Notch-radius sweep + energy-removed curve. Too small leaves "
             "residual grid; too large deletes legitimate low-frequency content and "
             "blurs the text — removing MORE frequencies is not always better.")
    p.body(
        "Filter-parameter effect: an intermediate radius (~4–8 px) is best. Below "
        "that the noise survives; well above it the notches eat into genuine image "
        "frequencies, softening the recovered text. Applications of frequency-domain "
        "restoration: interference/carrier removal in communications, de-striping in "
        "remote sensing, surveillance image cleanup, and periodic-artifact "
        "suppression in MRI/medical imaging.")

    p.add_page()
    p.h1("Q1B — Sobel Edge Detection (Missing Boundaries)")
    p.h2("Methodology")
    p.body(
        "Edges are locations of large gradient magnitude |∇I|. Sobel estimates the "
        "two partials with kernels that combine a central difference with [1 2 1] "
        "smoothing:\n"
        "    Gx = [[−1,0,1],[−2,0,2],[−1,0,1]],   Gy = Gxᵀ.\n"
        "Convolving gives gx, gy; then magnitude = √(gx²+gy²) and direction = "
        "atan2(gy,gx). Large magnitude ⇒ a sudden intensity transition ⇒ an edge. "
        "Our manual reflect-padded convolution matches SciPy's convolve2d to "
        "0.0 error (validated in the notebook).\n\n"
        "Why noise hurts and smoothing helps: differentiation is a HIGH-PASS "
        "operation (response ∝ |ω|), so it amplifies high-frequency noise into "
        "spurious edges; a Gaussian blur is a LOW-PASS filter that suppresses that "
        "noise first. The cost is that fine/weak edges are also smoothed away.")
    p.figure("q1b_sobel.png",
             "Fig 6. Input, gx, gy, gradient magnitude, direction, and thresholded "
             "edge map. Sobel reveals the building/sculpture boundaries.")
    p.figure("q1b_noise.png",
             "Fig 7. Sobel run directly on inputs with increasing Gaussian noise — "
             "spurious edges proliferate as the high-pass derivative amplifies noise.")
    p.figure("q1b_smoothing.png",
             "Fig 8. The noise-vs-detail trade-off: pre-smoothing a noisy image with "
             "increasing σ removes false edges but progressively erases weak/fine ones.")
    p.figure("q1b_canny_bilateral.png",
             "Fig 9. Can weak edges be preserved? Canny (non-maximum suppression + "
             "hysteresis) keeps faint-but-connected edges that a single Sobel "
             "threshold misses; a bilateral pre-filter keeps boundaries sharper than "
             "Gaussian smoothing.")
    p.body(
        "Answer to 'can weak edges be preserved without excess noise?': only "
        "partially with a single global threshold — there is an inherent trade-off. "
        "The standard fix is Canny's non-maximum suppression plus hysteresis "
        "(dual-threshold) which keeps weak edges that connect to strong ones while "
        "rejecting isolated noise; an edge-preserving bilateral pre-filter helps too.")
    p.output(os.path.join(REPORT, "Q1_report.pdf"))
    print("Wrote report/Q1_report.pdf")


# =========================================================== Q2 report
def build_q2():
    p = Report("Q2 — The Midnight Episode (ECG Arrhythmia)")
    title_block(p, "Catching the Arrhythmia via normalized template correlation")
    p.body(
        "Real Holter data: patient_ecg.npy (N = 5000 samples) and template.npy "
        "(L = 200 samples), fs = 250 Hz. Every number below is computed in the "
        "notebook, not eyeballed.")

    p.h2("(a) Reading the signal")
    p.body("(i) duration = N/fs = 5000/250 = 20.0 s.\n"
           "(ii) heart rate = 60/0.8 = 75 bpm; samples/beat = 0.8·250 = 200.\n"
           "(iii) fundamental f0 = 1/0.8 = 1.25 Hz.")
    p.figure("q2_ecg_full.png", "Fig 1. The 20 s recording: regular beats then an irregular region.")

    p.h2("(b) Healthy heart in the frequency domain")
    p.body("(i) A (nearly) periodic signal has a DISCRETE LINE spectrum (harmonics "
           "of f0), not a smooth continuous curve.\n"
           "(ii) The sharp, narrow QRS spike drives the HIGH-frequency content "
           "(narrow in time ⇒ broadband in frequency, time–bandwidth principle); the "
           "broad P and T waves are inherently low-frequency.\n"
           "(iii) At 150 bpm the period halves to 0.4 s, so f0 = 2.5 Hz and the "
           "harmonic spacing (= f0) DOUBLES.")
    p.figure("q2_spectrum.png", "Fig 2. Healthy-segment spectrum: peaks at multiples of f0 = 1.25 Hz (red).")

    p.h2("(c) Cutting a heartbeat (windowing)")
    p.body("(i) Window width ≈ 200 samples (one beat period), placed to bracket a "
           "full {P,QRS,T} starting just before the P wave.\n"
           "(ii) 80 samples = too short: slices into the beat, clips QRS/T, and the "
           "hard rectangular edges cause spectral leakage. 600 samples = too long: "
           "captures fragments of neighbouring beats, contaminating the template.\n"
           "(iii) Same uncertainty trade-off as STFT windows: shorter = sharper time "
           "but poorer frequency resolution and here it risks cutting the beat — so "
           "'as short as possible' is not automatically best.")
    p.figure("q2_windows.png", "Fig 3. Template and the two bad window widths.")

    p.h2("(d) Matching the template (normalized correlation)")
    p.body("ρ(m) = Σ t[k]x[m+k] / (‖t‖‖x_m‖).\n"
           "(i) ρ ∈ [−1, 1] (Cauchy–Schwarz); +1 = perfect match.\n"
           "(ii) Normalization removes amplitude/baseline-wander sensitivity: a beat "
           "twice as tall but identically shaped keeps ρ = +1, while the unnormalized "
           "dot product would roughly double — wrongly conflating shape with size.\n"
           "(iii) An inverted beat gives ρ ≈ −1, far from the healthy +1, so it is "
           "trivially flagged by a threshold.")

    p.h2("(e) Onset detection & the spectrogram")
    p.body("(i) Rule: first beat with ρ < threshold (0.5). Too high ⇒ false alarms "
           "from normal variability; too low ⇒ misses mild morphology changes.\n"
           "(ii) Spectrogram: healthy region shows clean steady harmonic bands; the "
           "arrhythmia region shows smeared/broken/shifting bands and new irregular "
           "energy.\n"
           "(iii) They can disagree because the spectrogram must accumulate several "
           "irregular beats in its window before structure visibly degrades (delayed, "
           "smeared), whereas correlation flags per beat — so trust CORRELATION for "
           "the exact onset, spectrogram for corroboration.")
    p.figure("q2_correlation.png",
             "Fig 4. ECG with detected onset (top) and ρ vs time (bottom). The first "
             "12 beats score ≈ +1; the arrhythmia onset at t = 9.6 s (sample 2400) "
             "begins with an inverted beat (ρ ≈ −1).")

    p.h2("(f) Sampling & aliasing")
    p.body("(i) Nyquist minimum = 2·40 = 80 Hz.\n"
           "(ii) At 50 Hz (< 80 Hz) the QRS content above 25 Hz aliases down, "
           "distorting the sharp spike — dangerous because the detector relies on "
           "shape correlation, so aliasing causes false positives/negatives.\n"
           "(iii) Fix: anti-alias low-pass (~25 Hz cutoff) before downsampling; "
           "unavoidable cost = losing genuine fine QRS diagnostic detail.")

    p.h2("(g) The detector — find_onset")
    p.body("find_onset(ecg, template, threshold) steps beat-by-beat (stride = "
           "len(template) = 200), scores each beat by normalized correlation, and "
           "returns the first index with ρ < threshold (else −1). On the provided "
           "data with threshold 0.5 it returns sample 2400 (t = 9.6 s). Implementation "
           "in src/q2_ecg.py.")

    p.h2("(h) Spectrogram window length")
    p.body("Frequency resolution Δf = fs/nperseg. To resolve the f0 = 1.25 Hz "
           "harmonic spacing we need Δf ≲ 1.25 Hz ⇒ nperseg ≳ 250/1.25 = 200. We use "
           "nperseg = 500 (Δf = 0.5 Hz) for clean, well-separated harmonic bands.")
    p.figure("q2_spectrogram.png",
             "Fig 5. Spectrogram (nperseg = 500): steady harmonic bands break up at "
             "the arrhythmia onset (cyan line), corroborating the correlation result.")
    p.output(os.path.join(REPORT, "Q2_report.pdf"))
    print("Wrote report/Q2_report.pdf")


# =========================================================== Q3 report
def build_q3():
    p = Report("Q3 — Sonic Signatures + App")
    title_block(p, "Q3A Audio fingerprinting + Q3B Streamlit identifier app")
    p.body(
        "DATA: the real 50-song library (Beatles / Queen, etc.) is indexed from "
        "data/songs/. Each song's filename without extension is its ground-truth "
        "label, as required, and filenames are kept exactly as provided. The whole "
        "library produces ~2.47 million paired hashes.")

    p.h1("Q3A — Sonic Signatures")
    p.h2("1. Why one DFT is not enough")
    p.body("The DFT of a whole song shows WHICH frequencies occur but not WHEN — all "
           "timing is lost, which is fatal for recognition.")
    p.figure("q3_dft.png", "Fig 1. Whole-song DFT magnitude: frequencies without time.")
    p.h2("2. Spectrogram & the time–frequency trade-off")
    p.body("A spectrogram takes the DFT of short sliding windows. Short window ⇒ "
           "sharp time / blurry frequency; long window ⇒ blurry time / sharp "
           "frequency. Same uncertainty trade-off as Q2(c)(iii).")
    p.figure("q3_window_tradeoff.png", "Fig 2. Short vs long STFT window.")
    p.h2("3. Constellation of peaks")
    p.body("We keep 2-D local maxima above an amplitude floor "
           "(scipy.ndimage.maximum_filter). These sparse high-energy points are "
           "noise-robust and form the song's constellation.")
    p.figure("q3_constellation.png", "Fig 3. Peaks overlaid on the spectrogram.")
    p.h2("4–5. Hashing & matching")
    p.body("Each anchor peak is paired with forward peers in a target zone into a "
           "hash (f1, f2, Δt) → (song, t). To match a query we accumulate "
           "offset = t_db − t_query per song; the TRUE song aligns all its hits at "
           "one offset (a sharp spike) while wrong songs scatter.")
    p.figure("q3_offset_hist.png",
             "Fig 4. Offset histograms — correct song: one sharp spike; wrong song: "
             "scattered noise. This is the deciding vote.")
    p.h2("6. Single peaks vs paired hashes")
    p.body("Matching on single peak frequencies gives a flat, noisy histogram even "
           "for the correct song, because a lone frequency bin recurs across "
           "unrelated songs (low specificity). Pairing constrains (f1, f2, Δt) "
           "jointly — far rarer to collide on — so the correct match becomes decisive "
           "(much higher winner-vs-runner-up ratio).")
    p.h2("7. Robustness")
    p.figure("q3_noise_robustness.png",
             "Fig 5. Match score vs SNR: recognition survives heavy additive noise "
             "then collapses past a breakdown SNR (peaks are robust until buried).")
    p.body("Pitch shift: even a small shift DEFEATS the identifier. The hashes key on "
           "ABSOLUTE frequency bins; a pitch shift multiplies every frequency by a "
           "constant, moving every peak to a new bin so essentially no hash collides "
           "— even though a human, who tracks RELATIVE pitch, hears the same song.\n"
           "Fix: quantize peaks to LOG-frequency / constant-Q bins (a pitch shift "
           "becomes a constant additive offset there), or allow a small ± tolerance "
           "in hash lookup, or pre-index a few pitch-shifted copies offline.")

    p.add_page()
    p.h1("Q3B — Web App: EE200 Audio Fingerprinting")
    p.body(
        "A custom dark-themed web app (Flask backend + HTML/CSS/JS frontend) that "
        "reuses the engine in src/q3_fingerprint.py — no duplicated logic. Code in "
        "app/web/. Verified end-to-end on the real 50-song library: all built-in "
        "sample clips identified correctly with very high confidence, and the batch "
        "endpoint returns the exact results.csv format.")
    p.h2("Required modes")
    p.body("• Single-clip (Identify tab): upload OR pick a sample, then see, in "
           "order, the pipeline-timing breakdown, the predicted song with a "
           "confidence (winner/runner-up) score, the candidate ranking, and the "
           "spectrogram -> constellation -> alignment-spike visualizations.\n"
           "• Batch tab: upload many clips -> results table -> Download results.csv "
           "with EXACTLY two columns filename,prediction (prediction = matched "
           "track's filename without extension, or 'none' below threshold).\n"
           "• Library tab: the full indexed library as cards (constellation "
           "thumbnail + hash count).")
    p.h2("Extra analysis built in (beyond the brief)")
    p.body("• Live microphone capture (encoded to WAV in-browser) for real "
           "Shazam-style identification.\n"
           "• Robustness Lab: re-tests the same clip across a white-noise SNR sweep "
           "and a pitch-shift sweep, charting how the match score survives heavy "
           "noise but collapses under even a +0.25-semitone shift (absolute-frequency "
           "vulnerability) — the Q3A robustness analysis, made interactive.\n"
           "• Single-peak vs paired-hash comparison on the live query: side-by-side "
           "offset histograms showing pairing is ~10x more decisive (e.g. 516x vs "
           "30x confidence on the same clip) — directly answering Q3A part 6.")
    p.h2("Deployment (to complete before submission)")
    p.body("The indexed DB ships in the repo (app/web/static/fingerprint_db.pkl), so "
           "the app runs immediately; the raw mp3s are not needed at runtime. See "
           "app/web/DEPLOY.md.\n"
           "1. Push the repo to GitHub.\n"
           "2. Deploy on Render (render.yaml is included) or Hugging Face Spaces "
           "(Dockerfile in DEPLOY.md). Streamlit Cloud cannot host a Flask app.\n"
           "3. Test the LIVE link end-to-end — the rubric gives 0 for Q3B if the "
           "link does not work.\n\n"
           "LIVE APP LINK:  __________________________  (paste after deploying)\n"
           "SOURCE CODE LINK: https://github.com/Mayurdp2403/ee200-course-project")
    p.output(os.path.join(REPORT, "Q3_report.pdf"))
    print("Wrote report/Q3_report.pdf")


if __name__ == "__main__":
    build_q1()
    build_q2()
    build_q3()
    print("All reports written to", REPORT)
    if _LEFTOVER:
        print("WARNING unmapped non-ASCII codepoints (rendered as '?'):",
              sorted(hex(ord(c)) for c in _LEFTOVER))
