'use strict';

/* ─── State ──────────────────────────── */
const S = { loading: false, split: false, docCount: 0, chunkCount: 0 };

/* ─── DOM ────────────────────────────── */
const $ = id => document.getElementById(id);
const D = {
    statusGlow:   $('statusGlow'),
    statusLabel:  $('statusLabel'),
    stage:        $('stage'),
    pdfInput:     $('pdfInput'),
    imgInput:     $('imgInput'),
    pdfBtn:       $('pdfBtn'),
    imgBtn:       $('imgBtn'),
    pdfBar:       $('pdfBar'),
    imgBar:       $('imgBar'),
    fileChips:    $('fileChips'),
    qInput:       $('qInput'),
    sendBtn:      $('sendBtn'),
    docsCountTxt: $('docsCountTxt'),
    answerScroll: $('answerScroll'),
    aLoading:     $('aLoading'),
    dotsLabel:    $('dotsLabel'),
    aContent:     $('aContent'),
    aQ:           $('aQ'),
    aBody:        $('aBody'),
    evidenceWrap: $('evidenceWrap'),
    evBadge:      $('evBadge'),
    evList:       $('evList'),
    toast:        $('toast'),
};

/* ─── Boot ───────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
    // Call /reset first — wipes server-side DB + uploads, then fetch fresh count
    await resetSession();
    initUploads();
    initTextarea();
    startHeartbeat();
});

/* ── Reset session on every page load ── */
async function resetSession() {
    try {
        await fetch('/api/v1/reset', { method: 'POST' });
    } catch (e) {
        // Server may not be up yet — ignore
    }
    // Always sync count after reset (should be 0)
    await loadDocCount();
}

/* ════════════════════════════════════════
   UPLOADS
   ════════════════════════════════════════ */
function initUploads() {
    D.pdfInput.addEventListener('change', e => { processFiles(e.target.files, 'pdf'); e.target.value = ''; });
    D.imgInput.addEventListener('change', e => { processFiles(e.target.files, 'img'); e.target.value = ''; });
    setupDrop(D.pdfBtn, f => processFiles(f, 'pdf'));
    setupDrop(D.imgBtn, f => processFiles(f, 'img'));
    setupDrop(document.body, files => Array.from(files).forEach(f =>
        processFiles([f], f.name.split('.').pop().toLowerCase() === 'pdf' ? 'pdf' : 'img')
    ));
}

function setupDrop(el, cb) {
    el.addEventListener('dragover', e => e.preventDefault());
    el.addEventListener('drop',     e => { e.preventDefault(); if (e.dataTransfer.files.length) cb(e.dataTransfer.files); });
}

async function processFiles(files, type) {
    for (const f of Array.from(files)) await uploadOne(f, type);
}

async function uploadOne(file, type) {
    const bar = type === 'pdf' ? D.pdfBar : D.imgBar;
    const btn = type === 'pdf' ? D.pdfBtn : D.imgBtn;
    const cid = 'c' + Date.now() + Math.random().toString(36).slice(2, 5);

    addChip(cid, file.name, 'working');
    btn.classList.add('uploading');
    setBar(bar, 10);

    try {
        const fd = new FormData();
        fd.append('file', file);
        setBar(bar, 65);

        // No timeout — large PDFs can take minutes to embed
        const res = await fetch('/api/v1/upload', { method: 'POST', body: fd });
        setBar(bar, 95);

        if (!res.ok) {
            const e = await res.json().catch(() => ({}));
            throw new Error(e.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        setBar(bar, 100);
        updateChip(cid, 'done', file.name);  // store original filename on chip
        btn.classList.remove('uploading');
        btn.classList.add('done');
        setTimeout(() => btn.classList.remove('done'), 2000);

        await loadDocCount();
        setStatus('ready');
        toast(`✓ ${file.name}  —  ${data.chunks_created} chunks`, 'ok');

    } catch (err) {
        setBar(bar, 0);
        btn.classList.remove('uploading');
        updateChip(cid, 'err');
        setStatus('error');
        toast(`✗ ${file.name}: ${err.message}`, 'err');
    } finally {
        setTimeout(() => setBar(bar, 0), 900);
    }
}

function setBar(b, pct) { b.style.width = pct + '%'; }

/* ════════════════════════════════════════
   CHIPS  (session visual only)
   Removing a chip also clears the entire DB
   (since we don't track per-file IDs in ChromaDB)
   ════════════════════════════════════════ */
function addChip(id, name, status) {
    const short = name.length > 28 ? name.slice(0, 25) + '…' : name;
    const el = document.createElement('div');
    el.className = `chip ${status}`; el.id = id;
    el.dataset.source = name;   // store original filename for per-doc delete
    el.innerHTML = `<span class="chip-name" title="${esc(name)}">${esc(short)}</span>
        <button class="chip-x" onclick="removeChip('${id}')" title="Remove">✕</button>`;
    D.fileChips.appendChild(el);
}
function updateChip(id, s, source) {
    const el = $(id);
    if (!el) return;
    el.className = `chip ${s}`;
    if (source) el.dataset.source = source;
}

async function removeChip(id) {
    const el = $(id);
    if (!el) return;
    const source = el.dataset.source || '';

    // Animate out immediately
    el.style.transition = 'opacity 0.17s, transform 0.17s';
    el.style.opacity = '0'; el.style.transform = 'scale(0.8)';
    setTimeout(() => el.remove(), 170);

    // Delete just this source from the vector store
    if (source) {
        try {
            await fetch('/api/v1/documents/' + encodeURIComponent(source), { method: 'DELETE' });
        } catch {}
    }
    await loadDocCount();
}

/* ════════════════════════════════════════
   TEXTAREA  — auto-resize
   ════════════════════════════════════════ */
function initTextarea() {
    const box = $('askBox');
    D.qInput.addEventListener('input', () => {
        // contenteditable: check if content exceeds ~2 lines
        const lineH = parseFloat(getComputedStyle(D.qInput).lineHeight) || 24;
        box.classList.toggle('tall', D.qInput.scrollHeight > lineH * 2.5);
    });
    // Prevent Enter from inserting <div> or <br> — submit instead
    D.qInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuestion();
        }
    });
}

/* ════════════════════════════════════════
   SUBMIT
   ════════════════════════════════════════ */
async function submitQuestion() {
    // Button is disabled when docCount=0, but double-check
    if (S.docCount === 0) { toast('Upload a document first', 'inf'); return; }
    const q = (D.qInput.innerText || D.qInput.textContent || '').trim();
    if (!q) { toast('Please enter a question', 'inf'); D.qInput.focus(); return; }
    if (S.loading) return;

    S.loading = true;
    setBusy(true);
    setStatus('loading');

    // Trigger the slide transition
    if (!S.split) {
        S.split = true;
        // Double rAF ensures browser has painted before transition fires
        requestAnimationFrame(() => requestAnimationFrame(() => D.stage.classList.add('split')));
    }

    showLoading();

    const stages = ['Searching documents…', 'Generating embeddings…', 'Reranking results…', 'Composing answer…'];
    let si = 0;
    const iv = setInterval(() => { si = (si + 1) % stages.length; D.dotsLabel.textContent = stages[si]; }, 1800);

    try {
        const res = await fetch('/api/v1/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, top_k: 5 }),
        });
        clearInterval(iv);

        if (!res.ok) {
            const e = await res.json().catch(() => ({}));
            throw new Error(e.detail || `HTTP ${res.status}`);
        }

        showAnswer(await res.json());
        setStatus('ready');

    } catch (err) {
        clearInterval(iv);
        showError(err.message);
        setStatus('error');
        toast('Request failed — check terminal', 'err');
    } finally {
        S.loading = false;
        setBusy(false);
    }
}

/* ════════════════════════════════════════
   ANSWER PANEL
   ════════════════════════════════════════ */
function showLoading() {
    D.aContent.style.display = 'none';
    D.aLoading.style.display = 'flex';
    D.dotsLabel.textContent  = 'Searching documents…';
    D.answerScroll.scrollTop = 0;
}

function showAnswer(data) {
    D.aLoading.style.display = 'none';
    D.aContent.style.display = 'flex';
    D.aQ.textContent         = `"${data.question}"`;
    D.aBody.innerHTML        = renderAnswer(data.answer);

    const unique = dedup(data.sources || []);
    if (unique.length > 0) {
        renderEvidence(unique);
        D.evidenceWrap.style.display = 'flex';
    } else {
        D.evidenceWrap.style.display = 'none';
    }
    D.answerScroll.scrollTop = 0;
}

function showError(msg) {
    D.aLoading.style.display = 'none';
    D.aContent.style.display = 'flex';
    D.aQ.textContent  = 'Error';
    D.aBody.innerHTML = `<p>⚠ ${esc(msg)}</p><p>Make sure Ollama is running:<br><code>ollama serve</code></p>`;
    D.evidenceWrap.style.display = 'none';
}

/* ════════════════════════════════════════
   ANSWER RENDERER
   ════════════════════════════════════════ */
function renderAnswer(text) {
    if (!text) return '';
    const lines = text.split('\n');
    let html = '', list = '';
    const closeList = () => { if (list) { html += `</${list}>`; list = ''; } };

    for (const raw of lines) {
        if (/^#{2,3}\s+/.test(raw)) {
            closeList(); html += `<h3>${inlineFmt(raw.replace(/^#{2,3}\s+/, ''))}</h3>`;
        } else if (/^[-*•]\s+/.test(raw)) {
            if (list !== 'ul') { closeList(); html += '<ul>'; list = 'ul'; }
            html += `<li>${inlineFmt(raw.replace(/^[-*•]\s+/, ''))}</li>`;
        } else if (/^\d+[.)]\s+/.test(raw)) {
            if (list !== 'ol') { closeList(); html += '<ol>'; list = 'ol'; }
            html += `<li>${inlineFmt(raw.replace(/^\d+[.)]\s+/, ''))}</li>`;
        } else if (raw.trim() === '') {
            closeList();
        } else {
            closeList(); html += `<p>${inlineFmt(raw)}</p>`;
        }
    }
    closeList();
    return html;
}
function inlineFmt(t) {
    return esc(t)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/`(.+?)`/g, '<code>$1</code>');
}

/* ════════════════════════════════════════
   EVIDENCE
   ════════════════════════════════════════ */
function dedup(src) {
    const ids = new Set(), txts = new Set();
    return src.filter(s => {
        const id  = s.chunk_id || '';
        const pre = (s.text || '').slice(0, 80).toLowerCase().trim();
        if (id && ids.has(id))  return false;
        if (txts.has(pre))      return false;
        if (id) ids.add(id);
        txts.add(pre);
        return true;
    });
}

function renderEvidence(sources) {
    D.evList.innerHTML = '';
    D.evBadge.textContent = `${sources.length}`;

    sources.forEach((src, i) => {
        const pct   = Math.round((src.score || 0) * 100);
        const name  = src.source || 'Unknown';
        const short = name.length > 42 ? '…' + name.slice(-39) : name;
        const page  = src.page;

        const card = document.createElement('div');
        card.className = 'ev-card';
        card.innerHTML = `
            <div class="ev-meta">
                <span class="ev-dot"></span>
                <span class="ev-name" title="${esc(name)}">${esc(short)}</span>
                <span class="ev-pct">${pct}%</span>
            </div>
            ${page ? `<div class="ev-page-ref">Page No: ${esc(String(page))}</div>` : ''}
            <div class="ev-text" id="evt${i}">${esc(src.text || '')}</div>
            <button class="ev-toggle" id="etg${i}" onclick="toggleEv(${i})">▸ Show more</button>
        `;
        D.evList.appendChild(card);
    });
}

function toggleEv(i) {
    const txt = $('evt' + i);
    const btn = $('etg' + i);
    if (!txt || !btn) return;
    const open = txt.classList.toggle('expanded');
    btn.textContent = open ? '▾ Show less' : '▸ Show more';
}

/* ════════════════════════════════════════
   DOC COUNT
   ════════════════════════════════════════ */
async function loadDocCount() {
    try {
        const res = await fetch('/api/v1/documents');
        if (!res.ok) return;
        const data = await res.json();
        S.docCount   = Array.isArray(data.documents) ? data.documents.length : 0;
        S.chunkCount = typeof data.total_chunks === 'number' ? data.total_chunks : 0;
        syncCount();
        syncSendBtn();
    } catch { /* server not ready */ }
}

function syncCount() {
    const base = `${S.docCount} document${S.docCount !== 1 ? 's' : ''} indexed`;
    D.docsCountTxt.textContent = S.chunkCount > 0
        ? `${base}  |  ${S.chunkCount} chunks`
        : base;
}

function syncSendBtn() {
    // Disable send button when no documents are indexed
    D.sendBtn.disabled = (S.docCount === 0);
}

/* ════════════════════════════════════════
   HEARTBEAT
   ════════════════════════════════════════ */
function startHeartbeat() {
    async function ping() {
        if (!navigator.onLine) { setStatus('offline'); return; }
        try {
            const res = await fetch('/health', { cache: 'no-store', signal: AbortSignal.timeout(3000) });
            if (res.ok) { if (!S.loading) setStatus('ready'); } else setStatus('offline');
        } catch { setStatus('offline'); }
    }
    window.addEventListener('online',  () => { if (!S.loading) ping(); });
    window.addEventListener('offline', () => setStatus('offline'));
    ping();
    setInterval(ping, 5000);
}

/* ════════════════════════════════════════
   STATUS / BUSY
   ════════════════════════════════════════ */
function setStatus(s) {
    if (s === 'ready' && S.loading) return;
    D.statusGlow.className = 'status-glow';
    if (s === 'loading')                       D.statusGlow.classList.add('busy');
    else if (s === 'error' || s === 'offline') D.statusGlow.classList.add('offline');
    const lbl = { loading:'Processing', error:'Error', offline:'Offline', ready:'Ready' };
    D.statusLabel.textContent = lbl[s] || 'Ready';
    if (s === 'error') setTimeout(() => { if (!S.loading) setStatus('ready'); }, 3000);
}

function setBusy(b) {
    // Keep disabled if no docs, otherwise follow loading state
    D.sendBtn.disabled = b || (S.docCount === 0);
    D.sendBtn.classList.toggle('loading', b);
}

/* ════════════════════════════════════════
   TOAST
   ════════════════════════════════════════ */
let _tt;
function toast(msg, type = 'inf') {
    clearTimeout(_tt);
    D.toast.textContent = msg;
    D.toast.className   = `toast show ${type}`;
    _tt = setTimeout(() => D.toast.classList.remove('show'), 3500);
}

function esc(s) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(String(s)));
    return d.innerHTML;
}