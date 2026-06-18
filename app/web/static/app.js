'use strict';

const $ = (id) => document.getElementById(id);
const spinner = $('spinner');
const showSpin = (t) => { $('spinner-text').textContent = t || 'Working…'; spinner.classList.remove('hidden'); };
const hideSpin = () => spinner.classList.add('hidden');

// ---------------- tabs ----------------
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    $(btn.dataset.tab).classList.add('active');
  });
});

// ---------------- library ----------------
async function loadLibrary() {
  const res = await fetch('/api/library');
  const data = await res.json();
  $('lib-count').textContent = `— ${data.count} tracks`;
  const grid = $('library-grid');
  grid.innerHTML = '';
  data.songs.forEach((s) => {
    const card = document.createElement('div');
    card.className = 'song-card';
    card.innerHTML = `
      <img src="/static/${s.thumb}" alt="${s.label}" loading="lazy" />
      <div class="info">
        <div class="title" title="${s.label}">${s.label}</div>
        <div class="hashes">${s.hashes.toLocaleString()} hashes</div>
      </div>`;
    grid.appendChild(card);
  });
}

// ---------------- samples ----------------
async function loadSamples() {
  const res = await fetch('/api/samples');
  const { samples } = await res.json();
  const box = $('samples');
  box.innerHTML = '';
  if (!samples.length) { box.innerHTML = '<p class="dim">No sample clips available.</p>'; return; }
  samples.forEach((s) => {
    const row = document.createElement('div');
    row.className = 'sample-row';
    row.innerHTML = `
      <span class="name">${s.name}</span>
      <audio controls preload="none" src="/static/${s.file}"></audio>
      <button class="try" data-sample="${s.name}">Try</button>`;
    box.appendChild(row);
  });
  box.querySelectorAll('.try').forEach((b) => {
    b.addEventListener('click', () => identifySample(b.dataset.sample));
  });
}

// ---------------- identify ----------------
const clipInput = $('clip-input');
const identifyBtn = $('identify-btn');
let chosenFile = null;
let lastQuery = null;   // {file} or {sample} — reused by compare / robustness

clipInput.addEventListener('change', () => {
  chosenFile = clipInput.files[0] || null;
  $('dz-hint').textContent = chosenFile ? chosenFile.name : 'Drop / pick a file • WAV, MP3, FLAC, OGG, M4A';
  identifyBtn.disabled = !chosenFile;
});
setupDrag('dropzone', clipInput, () => clipInput.dispatchEvent(new Event('change')));

identifyBtn.addEventListener('click', () => {
  if (!chosenFile) return;
  lastQuery = { file: chosenFile };
  runIdentify();
});

function identifySample(name) {
  lastQuery = { sample: name };
  runIdentify();
}

function queryForm(extra) {
  const fd = new FormData();
  if (lastQuery && lastQuery.file) fd.append('clip', lastQuery.file);
  else if (lastQuery && lastQuery.sample) fd.append('sample', lastQuery.sample);
  if (extra) Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

async function runIdentify() {
  showSpin('Fingerprinting & matching…');
  try {
    const res = await fetch('/api/identify', { method: 'POST', body: queryForm() });
    if (!res.ok) throw new Error('server error');
    renderIdentify(await res.json());
    // reset the deeper-analysis panels for the new query
    $('compare-out').classList.add('hidden');
    $('robust-out').classList.add('hidden');
  } catch (e) {
    alert('Identification failed: ' + e.message);
  } finally {
    hideSpin();
  }
}

// ---------------- microphone recorder (client-side WAV) ----------------
const recordBtn = $('record-btn');
let recState = null;

function encodeWAV(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); ws(8, 'WAVE');
  ws(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([v], { type: 'audio/wav' });
}

recordBtn.addEventListener('click', async () => {
  if (recState) { stopRecording(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { alert('Microphone unavailable or permission denied.'); return; }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const mute = ctx.createGain(); mute.gain.value = 0;  // avoid speaker feedback
  const chunks = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
  recState = { stream, ctx, src, proc, chunks };
  recordBtn.classList.add('recording'); $('rec-label').textContent = 'Stop';
  recState.timer = setTimeout(stopRecording, 12000);  // auto-stop at 12 s
});

function stopRecording() {
  if (!recState) return;
  clearTimeout(recState.timer);
  const { stream, ctx, src, proc, chunks } = recState;
  proc.disconnect(); src.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const len = chunks.reduce((a, c) => a + c.length, 0);
  const data = new Float32Array(len); let off = 0;
  chunks.forEach((c) => { data.set(c, off); off += c.length; });
  const rate = ctx.sampleRate; ctx.close();
  recordBtn.classList.remove('recording'); $('rec-label').textContent = 'Record';
  recState = null;
  if (len < rate) { alert('Recording too short — hold a few seconds.'); return; }
  const file = new File([encodeWAV(data, rate)], 'microphone.wav', { type: 'audio/wav' });
  chosenFile = file; lastQuery = { file };
  $('dz-hint').textContent = '🎤 microphone.wav'; identifyBtn.disabled = false;
  runIdentify();
}

// ---------------- compare matchers ----------------
$('compare-btn').addEventListener('click', async () => {
  if (!lastQuery) return;
  showSpin('Comparing matchers…');
  try {
    const r = await (await fetch('/api/compare', { method: 'POST', body: queryForm() })).json();
    $('compare-out').classList.remove('hidden');
    const vc = (d) => d.top
      ? `<b>${d.top}</b> — score ${d.score.toLocaleString()}, confidence ${d.confidence === null ? '∞' : d.confidence + '×'}`
      : 'no match';
    $('paired-verdict').innerHTML = vc(r.paired);
    $('single-verdict').innerHTML = vc(r.single);
    $('img-paired').src = r.paired.hist;
    $('img-single').src = r.single.hist;
  } catch (e) { alert('Compare failed: ' + e.message); }
  finally { hideSpin(); }
});

// ---------------- robustness lab ----------------
$('robust-btn').addEventListener('click', async () => {
  if (!lastQuery) return;
  showSpin('Stress-testing (noise + pitch sweeps)…');
  try {
    const r = await (await fetch('/api/robustness', { method: 'POST', body: queryForm() })).json();
    if (r.error) throw new Error(r.error);
    $('robust-out').classList.remove('hidden');
    $('img-noise').src = r.noise_chart;
    $('img-pitch').src = r.pitch_chart;
  } catch (e) { alert('Robustness test failed: ' + e.message); }
  finally { hideSpin(); }
});

function renderIdentify(r) {
  $('identify-results').classList.remove('hidden');

  // pipeline
  const stages = [
    ['1', 'SPECTROGRAM', r.timings.spectrogram, `${r.spec_shape[0]}×${r.spec_shape[1]}`],
    ['2', 'CONSTELLATION', r.timings.constellation, `${r.n_peaks.toLocaleString()} peaks`],
    ['3', 'HASHING', r.timings.hashing, `${r.n_hashes.toLocaleString()} hashes`],
    ['4', 'DB LOOKUP', r.timings.lookup, `${r.n_tracks} tracks`],
    ['5', 'SCORING', r.timings.scoring, `offset ${r.best_offset}`],
  ];
  const circ = ['①', '②', '③', '④', '⑤'];
  $('pipeline').innerHTML =
    stages.map(([n, l, ms, sub]) =>
      `<div class="stage"><div class="n">${circ[n - 1]}</div>` +
      `<div class="lbl">${l}</div><div class="ms">${ms} ms</div><div class="sub">${sub}</div></div>`
    ).join('') +
    `<div class="total">total ${r.timings.total} ms</div>`;

  // match banner
  const banner = $('match-banner');
  if (r.prediction) {
    banner.classList.remove('nomatch');
    $('match-label').textContent = 'MATCH FOUND';
    $('match-title').textContent = r.prediction;
    const conf = r.confidence === null ? '∞' : `${r.confidence}×`;
    $('match-sub').innerHTML = `cluster score <b>${r.score.toLocaleString()}</b> &middot; <b>${conf}</b> the runner-up`;
  } else {
    banner.classList.add('nomatch');
    $('match-label').textContent = 'NO CONFIDENT MATCH';
    $('match-title').textContent = 'none';
    $('match-sub').innerHTML = `best guess <b>${r.raw_best || '—'}</b> scored only ${r.score} — below the confidence threshold`;
  }

  // candidates
  const max = r.candidates.length ? r.candidates[0].score : 1;
  $('candidates').innerHTML = r.candidates.map((c, i) => `
    <div class="cand ${i === 0 ? 'top' : ''}">
      <div class="bar-wrap"><div class="bar" style="width:${Math.max(6, 100 * c.score / max)}%"></div>
        <span class="cname">${c.label}</span></div>
      <div class="cscore">${c.score.toLocaleString()}</div>
    </div>`).join('');

  // viz
  $('img-spectrogram').src = r.spectrogram_img;
  $('img-constellation').src = r.constellation_img;
  $('img-histogram').src = r.histogram_img;

  $('identify-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------- batch ----------------
const batchInput = $('batch-input');
const batchBtn = $('batch-btn');
let batchFiles = [];
let batchRows = [];

batchInput.addEventListener('change', () => {
  batchFiles = Array.from(batchInput.files);
  renderChips();
  batchBtn.disabled = batchFiles.length === 0;
});
setupDrag('batch-dropzone', batchInput, () => batchInput.dispatchEvent(new Event('change')));

function renderChips() {
  $('batch-chips').innerHTML = batchFiles.map((f) => `
    <div class="chip"><span class="ic">♪</span>
      <div class="meta"><div class="fn">${f.name}</div>
        <div class="sz">${(f.size / 1e6).toFixed(1)} MB</div></div></div>`).join('');
}

batchBtn.addEventListener('click', async () => {
  if (!batchFiles.length) return;
  const fd = new FormData();
  batchFiles.forEach((f) => fd.append('clips', f));
  showSpin(`Identifying ${batchFiles.length} clips…`);
  try {
    const res = await fetch('/api/batch', { method: 'POST', body: fd });
    const data = await res.json();
    batchRows = data.rows;
    $('batch-results').classList.remove('hidden');
    $('batch-tbody').innerHTML = data.rows.map((r) => `
      <tr><td>${r.filename}</td>
      <td class="pred ${r.prediction === 'none' ? 'none' : ''}">${r.prediction}</td></tr>`).join('');
    $('batch-summary').textContent =
      `${data.matched} / ${data.total} clips matched to a track (${data.total - data.matched} returned none).`;
  } catch (e) {
    alert('Batch failed: ' + e.message);
  } finally {
    hideSpin();
  }
});

$('download-csv').addEventListener('click', () => {
  // Exact auto-graded format: header + filename,prediction rows.
  const csv = 'filename,prediction\n' +
    batchRows.map((r) => `${r.filename},${r.prediction}`).join('\n') + '\n';
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'results.csv'; a.click();
  URL.revokeObjectURL(a.href);
});

// ---------------- drag & drop helper ----------------
function setupDrag(zoneId, input, onDrop) {
  const z = $(zoneId);
  ['dragover', 'dragenter'].forEach((ev) =>
    z.addEventListener(ev, (e) => { e.preventDefault(); z.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    z.addEventListener(ev, () => z.classList.remove('drag')));
  z.addEventListener('drop', (e) => {
    e.preventDefault();
    input.files = e.dataTransfer.files;
    onDrop();
  });
}

// ---------------- init ----------------
loadLibrary();
loadSamples();
