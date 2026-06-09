/**
 * Voice Conversion System — Frontend App v2
 * ==========================================
 * Gestionează navigarea SPA, toate paginile, polling antrenare,
 * comparație radar chart și export CSV.
 */

const API = '';  // relative URL, serverul servește și frontend-ul

// ============================================================
// UTILITĂȚI
// ============================================================

function toast(msg, type = 'info', duration = 4000) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    document.getElementById('toast-container').appendChild(el);
    setTimeout(() => el.remove(), duration);
}

function showLoading(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hideLoading(id) { document.getElementById(id)?.classList.add('hidden'); }
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }

function formatTime(ms) {
    if (!ms) return '—';
    if (ms < 1000) return `${ms} ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

function formatSeconds(sec) {
    if (!sec) return '—';
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60), s = sec % 60;
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
}

async function apiFetch(endpoint, options = {}) {
    const resp = await fetch(API + endpoint, options);
    if (!resp.ok) {
        let err = `HTTP ${resp.status}`;
        try { const d = await resp.json(); err = d.detail || err; } catch (e) { }
        throw new Error(err);
    }
    return resp.json();
}

// ============================================================


const fileInputs = {
    'knn': document.getElementById('source-file-input'),
    'lvc': document.getElementById('lvc-source-file-input'),
    'fvc': document.getElementById('fvc-source-file-input'),
    'tts': document.getElementById('tts-source-file-input'),
    'mknn': document.getElementById('mknn-source-file-input')
};

const convertButtons = {
    'knn': document.getElementById('convert-btn'),
    'lvc': document.getElementById('lvc-convert-btn'),
    'fvc': document.getElementById('fvc-convert-btn'),
    'tts': document.getElementById('tts-convert-btn'),
    'mknn': document.getElementById('mknn-convert-btn')
};

const resultCards = {
    'knn': {
        card: document.getElementById('result-card'), info: document.getElementById('result-info'),
        original: document.getElementById('result-original'), converted: document.getElementById('result-converted'),
        loading: document.getElementById('convert-loading')
    },
    'lvc': {
        card: document.getElementById('lvc-result-card'), info: document.getElementById('lvc-result-info'),
        original: document.getElementById('lvc-result-original'), converted: document.getElementById('lvc-result-converted'),
        loading: document.getElementById('lvc-convert-loading')
    },
    'fvc': {
        card: document.getElementById('fvc-result-card'), info: document.getElementById('fvc-result-info'),
        original: document.getElementById('fvc-result-original'), converted: document.getElementById('fvc-result-converted'),
        loading: document.getElementById('fvc-convert-loading')
    },
    'tts': {
        card: document.getElementById('tts-result-card'), info: document.getElementById('tts-result-info'),
        original: document.getElementById('tts-result-original'), converted: document.getElementById('tts-result-converted'),
        loading: document.getElementById('tts-convert-loading')
    },
    'mknn': {
        card: document.getElementById('mknn-result-card'), info: document.getElementById('mknn-result-info'),
        original: document.getElementById('mknn-result-original'), converted: document.getElementById('mknn-result-converted'),
        loading: document.getElementById('mknn-convert-loading')
    }
};

const convertEndpoints = {
    'knn': '/api/convert/knn',
    'lvc': '/api/convert/lightvc',
    'fvc': '/api/convert/freevc',
    'tts': '/api/tts/clone',
    'mknn': '/api/convert/mknn'
};


const targetSelects = {
    'knn': document.getElementById('target-speaker-select'),
    'lvc': document.getElementById('lvc-target-speaker-select'),
    'fvc': document.getElementById('freevc-target-speaker-select'),
    'mknn': document.getElementById('mknn-target-speaker-select')
};

const topkSliders = {
    'knn': document.getElementById('topk-slider'),
    'mknn': document.getElementById('mknn-topk-slider')
};

const topkValues = {
    'knn': document.getElementById('topk-value'),
    'mknn': document.getElementById('mknn-topk-value')
};

// NAVIGARE SPA
// ============================================================

function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const page = document.getElementById(`page-${pageId}`);
    const nav = document.getElementById(`nav-${pageId}`);
    if (page) page.classList.add('active');
    if (nav) nav.classList.add('active');

    // Acțiuni la schimbarea paginii
    if (pageId === 'speakers') loadSpeakers();
    if (pageId === 'convert' || pageId === 'convert-lightvc' || pageId === 'convert-freevc' || pageId === 'compare' || pageId === 'voice-clone') loadSpeakersIntoSelects();
    if (pageId === 'training') initTrainingPage();
    if (pageId === 'history') loadHistory();
    if (pageId === 'compare') initComparePage();
    if (pageId === 'convert-lightvc') checkLightVCStatus();
}

document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    item.addEventListener('click', e => {
        e.preventDefault();
        navigateTo(item.dataset.page);
    });
});

// ============================================================
// SYSTEM INFO (dashboard)
// ============================================================

async function loadSystemInfo() {
    try {
        const info = await apiFetch('/api/system-info');
        document.getElementById('stat-speakers').textContent = info.speakers_registered ?? 0;
        document.getElementById('stat-conversions').textContent = info.total_conversions ?? 0;
        document.getElementById('stat-device').textContent =
            info.gpu_name !== 'N/A' ? info.gpu_name.split(' ').slice(-1)[0] : 'CPU';

        // LightVC status removed (model ascuns din UI)

        document.getElementById('status-text').textContent = 'Server activ';
    } catch (e) {
        document.getElementById('status-text').textContent = 'Server inactiv';
        document.querySelector('.status-dot').style.background = 'var(--red)';
    }
}

// ============================================================
// SPEAKERS MANAGEMENT
// ============================================================

let speakersList = [];

async function loadSpeakers() {
    try {
        const data = await apiFetch('/api/speakers');
        speakersList = data.speakers || [];
        renderSpeakers();
    } catch (e) {
        toast('Eroare la incarcarea vorbitorilor', 'error');
    }
}

function renderSpeakers() {
    const container = document.getElementById('speakers-list');
    if (!speakersList.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">👤</div><p>Nu există vorbitori înregistrați.</p></div>';
        return;
    }
    container.innerHTML = speakersList.map(sp => `
        <div class="speaker-card" id="spk-${sp.id}">
            <div class="speaker-card-header">
                <span class="speaker-card-name">👤 ${sp.name}</span>
                <span class="speaker-card-refs">${sp.num_references} fișiere</span>
            </div>
            ${sp.description ? `<div class="speaker-card-desc">${sp.description}</div>` : ''}
            <div class="speaker-card-date">${sp.added_date ? new Date(sp.added_date).toLocaleString('ro-RO') : ''}</div>
            <div class="speaker-card-actions">
                <button class="btn-delete-speaker" onclick="deleteSpeaker('${sp.id}', '${sp.name}')">🗑 Șterge</button>
            </div>
        </div>
    `).join('');
}

async function loadSpeakersIntoSelects() {
    try {
        const data = await apiFetch('/api/speakers');
        speakersList = data.speakers || [];
    } catch (e) { return; }

    const options = speakersList.length
        ? speakersList.map(sp => `<option value="${sp.id}">${sp.name} (${sp.num_references} ref)</option>`).join('')
        : '<option value="">— Niciun vorbitor —</option>';

    ['target-speaker-select', 'lvc-target-speaker-select', 'freevc-target-speaker-select', 'rvc-target-speaker-select', 'yourtts-target-speaker-select', 'cmp-speaker-select', 'clone-target-speaker-select', 'mknn-target-speaker-select', 'rvc-speaker-select'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const current = el.value;
        el.innerHTML = `<option value="">— Selectează vorbitor —</option>` + options;
        if (current) el.value = current;
    });
}

async function deleteSpeaker(id, name) {
    if (!confirm(`Șterge vorbitorul "${name}"?`)) return;
    try {
        await apiFetch(`/api/speakers/${id}`, { method: 'DELETE' });
        toast(`Vorbitor "${name}" șters.`, 'success');
        loadSpeakers();
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    }
}

// Add speaker form
const speakerZone = document.getElementById('speaker-upload-zone');
const speakerFilesInput = document.getElementById('speaker-files-input');
let selectedSpeakerFiles = [];

speakerZone.addEventListener('click', () => speakerFilesInput.click());
speakerZone.addEventListener('dragover', e => { e.preventDefault(); speakerZone.classList.add('drag-over'); });
speakerZone.addEventListener('dragleave', () => speakerZone.classList.remove('drag-over'));
speakerZone.addEventListener('drop', e => {
    e.preventDefault();
    speakerZone.classList.remove('drag-over');
    selectedSpeakerFiles = [...e.dataTransfer.files].filter(f => f.type.startsWith('audio/'));
    renderSelectedFiles();
});

speakerFilesInput.addEventListener('change', () => {
    selectedSpeakerFiles = [...speakerFilesInput.files];
    renderSelectedFiles();
});

function renderSelectedFiles() {
    const container = document.getElementById('selected-files-list');
    if (!selectedSpeakerFiles.length) { container.classList.add('hidden'); return; }
    container.classList.remove('hidden');
    container.innerHTML = selectedSpeakerFiles.map(f => `
        <span class="selected-file-tag">🎵 ${f.name}</span>
    `).join('');
    document.getElementById('add-speaker-btn').disabled = false;
}

document.getElementById('add-speaker-btn')?.addEventListener('click', async () => {
    const name = document.getElementById('speaker-name').value.trim();
    const description = document.getElementById('speaker-description').value.trim();
    if (!name) { toast('Introdu un nume pentru vorbitor.', 'warning'); return; }
    if (!selectedSpeakerFiles.length) { toast('Selectează cel puțin un fișier audio.', 'warning'); return; }

    const fd = new FormData();
    fd.append('name', name);
    fd.append('description', description);
    selectedSpeakerFiles.forEach(f => fd.append('files', f));

    try {
        document.getElementById('add-speaker-btn').disabled = true;
        const res = await apiFetch('/api/speakers', { method: 'POST', body: fd });
        toast(`Vorbitor "${name}" adăugat (${res.num_references} fișiere).`, 'success');
        document.getElementById('speaker-name').value = '';
        document.getElementById('speaker-description').value = '';
        selectedSpeakerFiles = [];
        document.getElementById('selected-files-list').classList.add('hidden');
        document.getElementById('add-speaker-btn').disabled = true;
        loadSpeakers();
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
        document.getElementById('add-speaker-btn').disabled = false;
    }
});

// ============================================================
// CONVERSIE kNN-VC
// ============================================================

let sourceFile = null;
let lastConversionId = null;

function setupUploadZone(zoneId, inputId, previewId, filenameId, audioId, onFile) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return; // Prevent crash if element is missing from HTML
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) processAudioFile(file, previewId, filenameId, audioId, onFile);
    });
    input.addEventListener('change', () => {
        if (input.files[0]) processAudioFile(input.files[0], previewId, filenameId, audioId, onFile);
    });
}

function processAudioFile(file, previewId, filenameId, audioId, onFile) {
    const url = URL.createObjectURL(file);
    document.getElementById(filenameId).textContent = `📁 ${file.name}`;
    document.getElementById(audioId).src = url;
    show(previewId);
    if (onFile) onFile(file);
}

setupUploadZone('source-upload-zone', 'source-file-input', 'source-preview',
    'source-filename', 'source-audio',
    file => { sourceFile = file; updateConvertBtn(); }
);

Object.keys(topkSliders).forEach(key => {
    if (topkSliders[key] && topkValues[key]) {
        topkSliders[key].addEventListener('input', (e) => {
            topkValues[key].textContent = e.target.value;
        });
    }
});
document.getElementById('target-speaker-select')?.addEventListener('change', updateConvertBtn);

function updateConvertBtn() {
    const btn = document.getElementById('convert-btn');
    btn.disabled = !sourceFile || !document.getElementById('target-speaker-select').value;
}

document.getElementById('convert-btn')?.addEventListener('click', async () => {
    const selects = [
        document.getElementById('target-speaker-select'),
        document.getElementById('lvc-target-speaker-select'),
        document.getElementById('freevc-target-speaker-select'),
        document.getElementById('cmp-speaker-select'),
        document.getElementById('mknn-target-speaker-select')
    ];
    const speakerId = document.getElementById('target-speaker-select').value;
    const topk = parseInt(document.getElementById('topk-slider').value);
    if (!sourceFile || !speakerId) return;

    showLoading('convert-loading');
    hide('result-card');

    try {
        const fd = new FormData();
        fd.append('source', sourceFile);
        fd.append('speaker_id', speakerId);
        fd.append('topk', topk);

        const res = await apiFetch('/api/convert', { method: 'POST', body: fd });
        lastConversionId = res.conversion_id;

        document.getElementById('result-original').src = res.source_url;
        document.getElementById('result-converted').src = res.output_url;
        document.getElementById('result-info').innerHTML = `
            <span class="result-badge">kNN-VC</span>
            <span class="result-badge" style="background:rgba(34,197,94,0.15);color:var(--green);border-color:rgba(34,197,94,0.3)">${formatTime(res.conversion_time_ms)}</span>
            <span class="result-badge" style="background:rgba(168,85,247,0.15);color:#d8b4fe;border-color:rgba(168,85,247,0.3)">→ ${res.target_speaker}</span>
            <span class="result-badge" style="background:rgba(255,255,255,0.05);color:var(--text-muted);border-color:var(--border)">k=${topk}</span>
        `;

        if (res.source_text) {
            document.getElementById('original-stt').textContent = `Transcripție: "${res.source_text}"`;
            document.getElementById('converted-stt').innerHTML = `Transcripție: "${res.converted_text}" <br><span style="color:#94a3b8;font-size:0.8rem">WER: ${(res.wer * 100).toFixed(1)}%</span>`;
        } else {
            document.getElementById('original-stt').textContent = '';
            document.getElementById('converted-stt').textContent = '';
        }

        hide('metrics-grid');
        show('result-card');
        toast('Conversie finalizată!', 'success');
        loadSystemInfo();
    } catch (e) {
        toast(`Eroare conversie: ${e.message}`, 'error');
    } finally {
        hideLoading('convert-loading');
    }
});

// Evaluare metrici
document.getElementById('evaluate-btn')?.addEventListener('click', async () => {
    if (!lastConversionId) return;
    document.getElementById('evaluate-btn').disabled = true;
    document.getElementById('evaluate-btn').textContent = '📊 Se evaluează...';

    try {
        const fd = new FormData();
        fd.append('conversion_id', lastConversionId);
        const res = await apiFetch('/api/evaluate', { method: 'POST', body: fd });

        renderMetricsGrid(res.metrics, 'metrics-grid');
        show('metrics-grid');
        toast('Metrici calculate!', 'success');
    } catch (e) {
        toast(`Evaluare eșuată: ${e.message}`, 'error');
    } finally {
        document.getElementById('evaluate-btn').disabled = false;
        document.getElementById('evaluate-btn').innerHTML = '<span class="btn-icon">📊</span><span>Evaluează Calitatea</span>';
    }
});

function renderMetricsGrid(metrics, containerId) {
    const container = document.getElementById(containerId);
    const metricDefs = [
        { key: 'mcd', label: 'MCD', unit: 'dB', direction: 'lower', good: 7 },
        { key: 'pesq', label: 'PESQ', unit: '', direction: 'higher', good: 3 },
        { key: 'speaker_similarity', label: 'Spk Sim.', unit: '', direction: 'higher', good: 0.7 },
        { key: 'f0_rmse', label: 'F0 RMSE', unit: 'Hz', direction: 'lower', good: 30 },
        { key: 'f0_pcc', label: 'F0 PCC', unit: '', direction: 'higher', good: 0.8 },
        { key: 'snr', label: 'SNR', unit: 'dB', direction: 'higher', good: 15 },
    ];

    container.innerHTML = metricDefs.map(def => {
        const m = metrics[def.key];
        const val = m?.value ?? m ?? 0;
        const isGood = def.direction === 'lower' ? val <= def.good : val >= def.good;
        const quality = (val === 0) ? '' : (isGood ? 'good' : 'bad');
        return `
            <div class="metric-card ${quality}">
                <div class="metric-card-value">${val.toFixed ? val.toFixed(3) : val}</div>
                <div class="metric-card-label">${def.label}</div>
                <div class="metric-card-unit">${def.unit} ${def.direction === 'lower' ? '↓' : '↑'}</div>
            </div>
        `;
    }).join('');
}

// ============================================================
// CONVERSIE LightVC
// ============================================================

let lvcSourceFile = null;

setupUploadZone('lvc-source-upload-zone', 'lvc-source-file-input', 'lvc-source-preview',
    'lvc-source-filename', 'lvc-source-audio',
    file => { lvcSourceFile = file; updateLvcConvertBtn(); }
);

document.getElementById('lvc-target-speaker-select')?.addEventListener('change', updateLvcConvertBtn);

function updateLvcConvertBtn() {
    document.getElementById('lvc-convert-btn').disabled =
        !lvcSourceFile || !document.getElementById('lvc-target-speaker-select').value;
}

async function checkLightVCStatus() {
    try {
        const status = await apiFetch('/api/lightvc/status');
        if (!status.is_trained) {
            show('lightvc-not-trained-warning');
        } else {
            hide('lightvc-not-trained-warning');
        }
    } catch (e) { /* ignore */ }
}

document.getElementById('lvc-convert-btn')?.addEventListener('click', async () => {
    const speakerId = document.getElementById('lvc-target-speaker-select').value;
    if (!lvcSourceFile || !speakerId) return;

    showLoading('lvc-convert-loading');
    hide('lvc-result-card');

    try {
        const fd = new FormData();
        fd.append('source', lvcSourceFile);
        fd.append('speaker_id', speakerId);

        const res = await apiFetch('/api/lightvc/convert', { method: 'POST', body: fd });

        document.getElementById('lvc-result-original').src = res.source_url;
        document.getElementById('lvc-result-converted').src = res.output_url;
        document.getElementById('lvc-result-info').innerHTML = `
            <span class="result-badge" style="background:rgba(251,146,60,0.15);color:var(--lvc-primary);border-color:rgba(251,146,60,0.3)">LightVC</span>
            <span class="result-badge" style="background:rgba(34,197,94,0.15);color:var(--green);border-color:rgba(34,197,94,0.3)">${formatTime(res.conversion_time_ms)}</span>
        `;

        if (res.source_text) {
            document.getElementById('lvc-original-stt').textContent = `Transcripție: "${res.source_text}"`;
            document.getElementById('lvc-converted-stt').innerHTML = `Transcripție: "${res.converted_text}" <br><span style="color:#94a3b8;font-size:0.8rem">WER: ${(res.wer * 100).toFixed(1)}%</span>`;
        } else {
            document.getElementById('lvc-original-stt').textContent = '';
            document.getElementById('lvc-converted-stt').textContent = '';
        }
        show('lvc-result-card');
        toast('Conversie LightVC finalizată!', 'success');
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    } finally {
        hideLoading('lvc-convert-loading');
    }
});

// ============================================================
// CONVERSIE FreeVC
// ============================================================

let freevcSourceFile = null;

setupUploadZone('freevc-source-upload-zone', 'freevc-source-file-input', 'freevc-source-preview',
    'freevc-source-filename', 'freevc-source-audio',
    file => { freevcSourceFile = file; updateFreevcConvertBtn(); }
);

document.getElementById('freevc-target-speaker-select')?.addEventListener('change', updateFreevcConvertBtn);

function updateFreevcConvertBtn() {
    document.getElementById('freevc-convert-btn').disabled =
        !freevcSourceFile || !document.getElementById('freevc-target-speaker-select').value;
}

document.getElementById('freevc-convert-btn')?.addEventListener('click', async () => {
    const speakerId = document.getElementById('freevc-target-speaker-select').value;
    if (!freevcSourceFile || !speakerId) return;

    showLoading('freevc-convert-loading');
    hide('freevc-result-card');

    try {
        const fd = new FormData();
        fd.append('source', freevcSourceFile);
        fd.append('speaker_id', speakerId);

        const res = await apiFetch('/api/freevc/convert', { method: 'POST', body: fd });

        document.getElementById('freevc-result-original').src = res.source_url;
        document.getElementById('freevc-result-converted').src = res.output_url;
        document.getElementById('freevc-result-info').innerHTML = `
            <span class="result-badge" style="background:rgba(59,130,246,0.15);color:#3b82f6;border-color:rgba(59,130,246,0.3)">FreeVC</span>
            <span class="result-badge" style="background:rgba(34,197,94,0.15);color:var(--green);border-color:rgba(34,197,94,0.3)">${formatTime(res.conversion_time_ms)}</span>
        `;

        // if (res.source_text) {
        ///       document.getElementById('freevc-original-stt').textContent = `Transcripție: "${res.source_text}"`;
        ///        document.getElementById('freevc-converted-stt').innerHTML = `Transcripție: "${res.converted_text}" <br><span style="color:#94a3b8;font-size:0.8rem">WER: ${(res.wer * 100).toFixed(1)}%</span>`;
        ////    } else {
        ///        document.getElementById('freevc-original-stt').textContent = '';
        ///         document.getElementById('freevc-converted-stt').textContent = '';
        //    }

        show('freevc-result-card');
        toast('Conversie FreeVC finalizată!', 'success');
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    } finally {
        hideLoading('freevc-convert-loading');
    }
});

// ============================================================
// TRAINING PAGE
// ============================================================

let lossChart = null;
let trainingPollInterval = null;

async function initTrainingPage() {
    await refreshTrainingStatus();
    // Poll progres dacă antrenarea e în curs
    startTrainingPolling();
}

async function refreshTrainingStatus() {
    try {
        const status = await apiFetch('/api/lightvc/status');
        const ckInfo = status.checkpoint_info;

        const isTraining = status.is_training;
        const isTrained = status.is_trained;
        const progress = status.training_progress;

        // Model status
        document.getElementById('ts-status').textContent =
            isTraining ? '🔄 Antrenare în curs...' :
                isTrained ? '✅ Antrenat' : '⭕ Neantrenat';
        document.getElementById('ts-status').style.color =
            isTraining ? 'var(--yellow)' : isTrained ? 'var(--green)' : 'var(--text-muted)';

        document.getElementById('ts-epoch').textContent =
            ckInfo?.epoch ? `${ckInfo.epoch}` : (progress?.epoch ? `${progress.epoch}/${progress.total_epochs}` : '—');
        document.getElementById('ts-val-loss').textContent =
            ckInfo?.best_val_loss?.toFixed(4) ?? (progress?.best_val_loss?.toFixed(4) ?? '—');
        document.getElementById('ts-time').textContent =
            ckInfo?.training_time_hours ? `${ckInfo.training_time_hours.toFixed(1)}h` : '—';
        document.getElementById('ts-speakers').textContent =
            ckInfo?.num_speakers ?? progress?.num_speakers ?? '—';

        // Butoane
        if (isTraining) {
            hide('start-train-btn');
            show('stop-train-btn');
            show('training-progress-card');
            document.getElementById('nav-training').querySelector('.nav-badge').style.display = '';
            updateProgressUI(progress);
        } else {
            show('start-train-btn');
            hide('stop-train-btn');
            document.getElementById('nav-training').querySelector('.nav-badge').style.display = 'none';
            if (progress?.status === 'done' || progress?.status === 'stopped') {
                show('training-progress-card');
                updateProgressUI(progress);
            }
        }
    } catch (e) {
        console.error('Training status error:', e);
    }
}

function updateProgressUI(progress) {
    if (!progress) return;
    const epoch = progress.epoch || 0;
    const total = progress.total_epochs || 1;
    const pct = Math.round((epoch / total) * 100);

    document.getElementById('prog-epoch-text').textContent = `Epoca ${epoch} / ${total}`;
    document.getElementById('prog-loss-text').textContent =
        `Train: ${progress.train_loss?.toFixed(4) ?? '—'}` +
        (progress.val_loss ? ` | Val: ${progress.val_loss.toFixed(4)}` : '');
    document.getElementById('prog-eta-text').textContent =
        progress.eta_seconds ? `ETA: ${formatSeconds(progress.eta_seconds)}` : '';
    document.getElementById('training-progress-bar').style.width = `${pct}%`;
    document.getElementById('prog-message').textContent = progress.message || '';

    // Update chart
    if (progress.train_loss_history?.length > 0) {
        updateLossChart(progress.train_loss_history, progress.val_loss_history || []);
    }
}

function updateLossChart(trainHistory, valHistory) {
    const ctx = document.getElementById('loss-chart');
    if (!ctx) return;

    const labels = trainHistory.map((_, i) => i + 1);

    if (lossChart) {
        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = trainHistory;
        lossChart.data.datasets[1].data = valHistory;
        lossChart.update('none');
        return;
    }

    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Train Loss',
                    data: trainHistory,
                    borderColor: 'rgba(34,211,238,0.8)',
                    backgroundColor: 'rgba(34,211,238,0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Val Loss',
                    data: valHistory,
                    borderColor: 'rgba(251,146,60,0.8)',
                    backgroundColor: 'rgba(251,146,60,0.05)',
                    borderWidth: 2,
                    pointRadius: 2,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 11 } } }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', maxTicksLimit: 10 },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });
}

function startTrainingPolling() {
    if (trainingPollInterval) clearInterval(trainingPollInterval);
    trainingPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (!page.classList.contains('active')) return;
        await refreshTrainingStatus();
    }, 3000);
}

document.getElementById('start-train-btn')?.addEventListener('click', async () => {
    const epochs = parseInt(document.getElementById('train-epochs').value);
    const speakers = parseInt(document.getElementById('train-speakers').value);
    const batch = parseInt(document.getElementById('train-batch').value);

    if (!confirm(`Pornești antrenarea LightVC?\n• ${epochs} epoci\n• ${speakers} pseudo-vorbitori\n• Batch size: ${batch}\n\nAntrenarea rulează în background și poate dura câteva ore.`)) return;

    try {
        const fd = new FormData();
        fd.append('epochs', epochs);
        fd.append('n_speakers', speakers);
        fd.append('batch_size', batch);

        await apiFetch('/api/lightvc/train', { method: 'POST', body: fd });
        toast('Antrenare pornită! Urmărește progresul mai jos.', 'success');
        show('training-progress-card');
        await refreshTrainingStatus();
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    }
});

document.getElementById('stop-train-btn')?.addEventListener('click', async () => {
    if (!confirm('Oprești antrenarea? Progresul curent va fi salvat.')) return;
    try {
        await apiFetch('/api/lightvc/train/stop', { method: 'POST' });
        toast('Semnal de oprire trimis.', 'warning');
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    }
});

// ============================================================
// COMPARATIE MODELE
// ============================================================

let cmpSourceFile = null;
let radarChart = null;
let lastCompareResult = null;

function initComparePage() {
    updateCompareBtn();
}

setupUploadZone('cmp-upload-zone', 'cmp-source-input', 'cmp-source-preview',
    'cmp-source-filename', 'cmp-source-audio',
    file => { cmpSourceFile = file; updateCompareBtn(); }
);

// 'cmp-source-filename' doesn't exist so we need to handle it
function setupUploadZone(zoneId, inputId, previewId, filenameId, audioId, onFile) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file) {
            const audioEl = document.getElementById(audioId);
            if (audioEl) audioEl.src = URL.createObjectURL(file);
            const filenameEl = document.getElementById(filenameId);
            if (filenameEl) filenameEl.textContent = `📁 ${file.name}`;
            if (previewId) show(previewId);
            if (onFile) onFile(file);
        }
    });
    input.addEventListener('change', () => {
        const file = input.files[0];
        if (file) {
            const audioEl = document.getElementById(audioId);
            if (audioEl) audioEl.src = URL.createObjectURL(file);
            const filenameEl = document.getElementById(filenameId);
            if (filenameEl) filenameEl.textContent = `📁 ${file.name}`;
            if (previewId) show(previewId);
            if (onFile) onFile(file);
        }
    });
}

document.getElementById('cmp-speaker-select')?.addEventListener('change', updateCompareBtn);

document.getElementById('cmp-topk-slider')?.addEventListener('input', e => {
    document.getElementById('cmp-topk-val').textContent = e.target.value;
});

function updateCompareBtn() {
    document.getElementById('compare-btn').disabled =
        !cmpSourceFile || !document.getElementById('cmp-speaker-select').value;
}

document.getElementById('compare-btn')?.addEventListener('click', async () => {
    const speakerId = document.getElementById('cmp-speaker-select').value;
    const topk = parseInt(document.getElementById('cmp-topk-slider').value);
    if (!cmpSourceFile || !speakerId) return;

    showLoading('compare-loading');
    hide('compare-results');

    try {
        const fd = new FormData();
        fd.append('source', cmpSourceFile);
        fd.append('speaker_id', speakerId);
        fd.append('topk', topk);

        const res = await apiFetch('/api/compare', { method: 'POST', body: fd });
        lastCompareResult = res;
        renderCompareResults(res);
        show('compare-results');
        toast('Comparație finalizată!', 'success');
    } catch (e) {
        toast(`Eroare: ${e.message}`, 'error');
    } finally {
        hideLoading('compare-loading');
    }
});

function renderCompareResults(res) {
    // Sursa STT
    const sttEl = document.getElementById('cmp-source-stt');
    if (res.source_text) {
        sttEl.textContent = `"${res.source_text}"`;
    } else {
        sttEl.textContent = '—';
    }

    if (res.source && res.source.output_url) {
        const srcPlayer = document.getElementById('cmp-source-audio');
        srcPlayer.src = res.source.output_url;
        srcPlayer.style.display = 'block';
    }

    // kNN-VC player
    const knnData = res.knn_vc;
    if (knnData?.status === 'success') {
        document.getElementById('cmp-knn-audio').src = knnData.output_url;
        document.getElementById('cmp-knn-time').textContent = `⏱ ${formatTime(knnData.conversion_time_ms)}`;
        if (knnData.converted_text) {
            // document.getElementById('cmp-knn-stt').innerHTML = `"${knnData.converted_text}"<br><span style="opacity:0.7">WER: ${(knnData.wer * 100).toFixed(1)}%</span>`;
        } else {
            document.getElementById('cmp-knn-stt').innerHTML = '';
        }
        renderCmpMetrics(knnData.metrics || {}, 'cmp-knn-metrics');
    }

    // YourTTS player
    const yourttsData = res.yourtts;
    if (yourttsData?.status === 'success') {
        document.getElementById('cmp-yourtts-audio').src = yourttsData.output_url;
        document.getElementById('cmp-yourtts-time').textContent = `⏱ ${formatTime(yourttsData.conversion_time_ms)}`;
        if (yourttsData.epoch) {
            document.getElementById('cmp-yourtts-time').textContent +=
                ` · Epoch ${yourttsData.epoch}`;
        }
        if (yourttsData.converted_text) {
            // document.getElementById('cmp-yourtts-stt').innerHTML = `"${yourttsData.converted_text}"<br><span style="opacity:0.7">WER: ${(yourttsData.wer * 100).toFixed(1)}%</span>`;
        } else {
            document.getElementById('cmp-yourtts-stt').innerHTML = '';
        }
        renderCmpMetrics(yourttsData.metrics || {}, 'cmp-yourtts-metrics');
    } else if (yourttsData?.status === 'not_trained') {
        document.getElementById('cmp-yourtts-metrics').innerHTML =
            `<div style="color:var(--text-muted);font-size:0.8rem;padding:12px">${yourttsData.message}</div>`;
        document.getElementById('cmp-yourtts-stt').innerHTML = '';
    } else {
        const err = yourttsData?.error || 'Eroare necunoscută';
        document.getElementById('cmp-yourtts-metrics').innerHTML =
            `<div style="color:var(--red);font-size:0.8rem;padding:12px">Eroare: ${err}</div>`;
        document.getElementById('cmp-yourtts-stt').innerHTML = '';
    }

    // FreeVC player
    const fvcData = res.freevc;
    if (fvcData?.status === 'success') {
        document.getElementById('cmp-freevc-audio').src = fvcData.output_url;
        document.getElementById('cmp-freevc-time').textContent = `⏱ ${formatTime(fvcData.conversion_time_ms)}`;
        if (fvcData.converted_text) {
            // document.getElementById('cmp-freevc-stt').innerHTML = `"${fvcData.converted_text}"<br><span style="opacity:0.7">WER: ${(fvcData.wer * 100).toFixed(1)}%</span>`;
        } else {
            document.getElementById('cmp-freevc-stt').innerHTML = '';
        }
        renderCmpMetrics(fvcData.metrics || {}, 'cmp-freevc-metrics');
    } else {
        const err = fvcData?.error || 'Eroare necunoscută';
        document.getElementById('cmp-freevc-metrics').innerHTML =
            `<div style="color:var(--red);font-size:0.8rem;padding:12px">Eroare: ${err}</div>`;
        document.getElementById('cmp-freevc-stt').innerHTML = '';
    }

    // Voice Clone player
    const cloneData = res.voice_clone;
    if (cloneData?.status === 'success') {
        document.getElementById('cmp-clone-audio').src = cloneData.output_url;
        document.getElementById('cmp-clone-time').textContent = `⏱ ${formatTime(cloneData.conversion_time_ms)}`;

        let tags = '';
        if (cloneData.is_finetuned) tags += '<span style="display:inline-block;background:var(--purple-light);color:var(--purple);padding:2px 6px;border-radius:4px;font-size:0.75rem;margin-right:4px;">FT RO</span>';
        if (cloneData.rvc_applied) tags += '<span style="display:inline-block;background:var(--orange-light);color:var(--orange);padding:2px 6px;border-radius:4px;font-size:0.75rem;">+ RVC</span>';

        if (cloneData.transcribed_text) {
            document.getElementById('cmp-clone-stt').innerHTML = `"${cloneData.transcribed_text}"<br><div style="margin-top:4px">${tags}</div>`;
        } else {
            document.getElementById('cmp-clone-stt').innerHTML = tags ? `<div style="margin-top:4px">${tags}</div>` : '';
        }
        renderCmpMetrics(cloneData.metrics || {}, 'cmp-clone-metrics');
    } else {
        const err = cloneData?.error || 'Eroare necunoscută';
        document.getElementById('cmp-clone-metrics').innerHTML =
            `<div style="color:var(--red);font-size:0.8rem;padding:12px">Eroare: ${err}</div>`;
        document.getElementById('cmp-clone-stt').innerHTML = '';
    }

    // Radar chart
    if (knnData?.metrics || yourttsData?.metrics || fvcData?.metrics || cloneData?.metrics) {
        renderRadarChart(knnData?.metrics, yourttsData?.metrics, fvcData?.metrics, cloneData?.metrics);
        renderMetricsTable(knnData?.metrics, yourttsData?.metrics, fvcData?.metrics, cloneData?.metrics);
    }
}

function renderCmpMetrics(metrics, containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const defs = [
        { key: 'mcd', label: 'MCD', dir: 'lower' },
        { key: 'speaker_similarity', label: 'Spk Sim', dir: 'higher' },
        { key: 'f0_pcc', label: 'F0 PCC', dir: 'higher' },
        { key: 'pesq', label: 'PESQ', dir: 'higher' },
        { key: 'snr', label: 'SNR', dir: 'higher' },
        { key: 'f0_rmse', label: 'F0 RMSE', dir: 'lower' },
    ];

    el.innerHTML = defs.map(def => {
        const val = metrics[def.key];
        const display = (val != null && !isNaN(val)) ? Number(val).toFixed(3) : '—';
        return `
            <div class="cmp-metric-item">
                <div class="cmp-metric-value">${display}</div>
                <div class="cmp-metric-label">${def.label}</div>
            </div>
        `;
    }).join('');
}

function highlightWinners(knnMetrics, lvcMetrics, fvcMetrics, cloneMetrics) {
    const defs = [
        { key: 'mcd', dir: 'lower' },
        { key: 'speaker_similarity', dir: 'higher' },
        { key: 'f0_pcc', dir: 'higher' },
        { key: 'pesq', dir: 'higher' },
        { key: 'snr', dir: 'higher' },
        { key: 'f0_rmse', dir: 'lower' },
    ];

    defs.forEach((def, i) => {
        const knnVal = knnMetrics?.[def.key];
        const lvcVal = lvcMetrics?.[def.key];
        const fvcVal = fvcMetrics?.[def.key];
        const cloneVal = cloneMetrics?.[def.key];

        let bestVal = def.dir === 'lower' ? Infinity : -Infinity;
        if (knnVal != null) bestVal = def.dir === 'lower' ? Math.min(bestVal, knnVal) : Math.max(bestVal, knnVal);
        if (lvcVal != null) bestVal = def.dir === 'lower' ? Math.min(bestVal, lvcVal) : Math.max(bestVal, lvcVal);
        if (fvcVal != null) bestVal = def.dir === 'lower' ? Math.min(bestVal, fvcVal) : Math.max(bestVal, fvcVal);
        if (cloneVal != null) bestVal = def.dir === 'lower' ? Math.min(bestVal, cloneVal) : Math.max(bestVal, cloneVal);

        const knnItems = document.querySelectorAll('#cmp-knn-metrics .cmp-metric-item');
        const lvcItems = document.querySelectorAll('#cmp-lvc-metrics .cmp-metric-item');
        const fvcItems = document.querySelectorAll('#cmp-freevc-metrics .cmp-metric-item');
        if (knnItems[i]) knnItems[i].classList.toggle('cmp-metric-winner', knnVal === bestVal && knnVal != null);
        if (lvcItems[i]) lvcItems[i].classList.toggle('cmp-metric-winner', lvcVal === bestVal && lvcVal != null);
        if (fvcItems[i]) fvcItems[i].classList.toggle('cmp-metric-winner', fvcVal === bestVal && fvcVal != null);
        if (cloneItems[i]) cloneItems[i].classList.toggle('cmp-metric-winner', cloneVal === bestVal && cloneVal != null);
    });
}

function renderRadarChart(knnMetrics, yourttsMetrics, fvcMetrics, cloneMetrics) {
    const ctx = document.getElementById('radar-chart');
    if (!ctx) return;

    function normalize(metrics) {
        if (!metrics) return [0, 0, 0, 0, 0, 0];
        const mcd = Math.max(0, 1 - (metrics.mcd || 0) / 15);
        const pesq = ((metrics.pesq || 0) + 0.5) / 5;
        const spk = metrics.speaker_similarity || 0;
        const f0pcc = (metrics.f0_pcc || 0);
        const snr = Math.min(1, Math.max(0, (metrics.snr || 0) / 30));
        const f0rmse = Math.max(0, 1 - (metrics.f0_rmse || 0) / 100);
        return [mcd, pesq, spk, f0pcc, snr, f0rmse];
    }

    const labels = ['MCD', 'PESQ', 'Spk Sim.', 'F0 PCC', 'SNR', 'F0 Stab.'];
    const knnNorm = normalize(knnMetrics);
    const yourttsNorm = normalize(yourttsMetrics);
    const fvcNorm = normalize(fvcMetrics);
    const cloneNorm = normalize(cloneMetrics);

    if (window.radarChart) {
        window.radarChart.destroy();
    }

    window.radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'kNN-VC',
                    data: knnNorm,
                    borderColor: 'rgba(99,102,241,0.9)',
                    backgroundColor: 'rgba(99,102,241,0.1)',
                    pointBackgroundColor: 'rgba(99,102,241,1)',
                    pointRadius: 4
                },
                {
                    label: 'YourTTS',
                    data: yourttsNorm,
                    borderColor: 'rgba(251,146,60,0.9)',
                    backgroundColor: 'rgba(251,146,60,0.1)',
                    pointBackgroundColor: 'rgba(251,146,60,1)',
                    pointRadius: 4
                },
                {
                    label: 'FreeVC',
                    data: fvcNorm,
                    borderColor: 'rgba(59,130,246,0.9)',
                    backgroundColor: 'rgba(59,130,246,0.1)',
                    pointBackgroundColor: 'rgba(59,130,246,1)',
                    pointRadius: 4
                },
                {
                    label: 'Voice Clone',
                    data: cloneNorm,
                    borderColor: 'rgba(239,68,68,0.9)',
                    backgroundColor: 'rgba(239,68,68,0.1)',
                    pointBackgroundColor: 'rgba(239,68,68,1)',
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(0)}%`
                    }
                }
            },
            scales: {
                r: {
                    min: 0, max: 1,
                    ticks: { display: false },
                    grid: { color: 'rgba(255,255,255,0.08)' },
                    angleLines: { color: 'rgba(255,255,255,0.08)' },
                    pointLabels: { color: '#94a3b8', font: { size: 11 } }
                }
            }
        }
    });

    // Evidențiere câștigători
    if (knnMetrics || yourttsMetrics || fvcMetrics || cloneMetrics) highlightWinners(knnMetrics, yourttsMetrics, fvcMetrics, cloneMetrics);
}

function highlightWinners(knnMetrics, yourttsMetrics, fvcMetrics, cloneMetrics) {
    const defs = [
        { key: 'mcd', dir: 'lower' },
        { key: 'pesq', dir: 'higher' },
        { key: 'speaker_similarity', dir: 'higher' },
        { key: 'f0_rmse', dir: 'lower' },
        { key: 'f0_pcc', dir: 'higher' },
        { key: 'snr', dir: 'higher' },
    ];

    defs.forEach((def, i) => {
        const knnVal = knnMetrics?.[def.key];
        const yVal = yourttsMetrics?.[def.key];
        const fvcVal = fvcMetrics?.[def.key];
        const cloneVal = cloneMetrics?.[def.key];

        const validVals = [knnVal, yVal, fvcVal, cloneVal].filter(v => v != null);
        if (validVals.length === 0) return;

        const bestVal = def.dir === 'lower' ? Math.min(...validVals) : Math.max(...validVals);

        const knnItems = document.querySelectorAll('#cmp-knn-metrics .cmp-metric-item');
        const yItems = document.querySelectorAll('#cmp-yourtts-metrics .cmp-metric-item');
        const fvcItems = document.querySelectorAll('#cmp-freevc-metrics .cmp-metric-item');
        const cloneItems = document.querySelectorAll('#cmp-clone-metrics .cmp-metric-item');

        if (knnItems[i]) knnItems[i].classList.toggle('cmp-metric-winner', knnVal === bestVal && knnVal != null);
        if (yItems[i]) yItems[i].classList.toggle('cmp-metric-winner', yVal === bestVal && yVal != null);
        if (fvcItems[i]) fvcItems[i].classList.toggle('cmp-metric-winner', fvcVal === bestVal && fvcVal != null);
        if (cloneItems[i]) cloneItems[i].classList.toggle('cmp-metric-winner', cloneVal === bestVal && cloneVal != null);
    });
}

function renderMetricsTable(knnMetrics, yourttsMetrics, fvcMetrics, cloneMetrics) {
    const container = document.getElementById('metrics-compare-table');
    if (!container) return;

    if (!knnMetrics && !yourttsMetrics && !fvcMetrics && !cloneMetrics) {
        container.innerHTML = '';
        return;
    }

    const rows = [
        { key: 'mcd', label: 'MCD (Mel Cepstral Dist.)', unit: 'dB', best: 'min' },
        { key: 'pesq', label: 'PESQ (Calitate Audio)', unit: '', best: 'max' },
        { key: 'speaker_similarity', label: 'Speaker Similarity', unit: '', best: 'max' },
        { key: 'f0_rmse', label: 'F0 RMSE (Stabilitate Ton)', unit: 'Hz', best: 'min' },
        { key: 'f0_pcc', label: 'F0 PCC (Corelație)', unit: '', best: 'max' },
        { key: 'snr', label: 'SNR (Signal-to-Noise)', unit: 'dB', best: 'max' }
    ].map(m => {
        const kVal = knnMetrics?.[m.key];
        const yVal = yourttsMetrics?.[m.key];
        const fVal = fvcMetrics?.[m.key];
        const cVal = cloneMetrics?.[m.key];

        const validVals = [kVal, yVal, fVal, cVal].filter(v => typeof v === 'number');
        let bestVal = null;
        if (validVals.length > 0) {
            bestVal = m.best === 'min' ? Math.min(...validVals) : Math.max(...validVals);
        }

        const format = (val) => {
            if (typeof val !== 'number') return '—';
            const v = val.toFixed(m.key === 'mcd' || m.key === 'snr' ? 1 : 2);
            return val === bestVal ? `<strong>${v}</strong>` : v;
        };

        return `<tr>
            <td style="text-align:left;font-weight:500">${m.label}</td>
            <td>${format(kVal)} ${kVal !== undefined ? m.unit : ''}</td>
            <td>${format(yVal)} ${yVal !== undefined ? m.unit : ''}</td>
            <td>${format(fVal)} ${fVal !== undefined ? m.unit : ''}</td>
            <td>${format(cVal)} ${cVal !== undefined ? m.unit : ''}</td>
        </tr>`;
    }).join('');

    container.innerHTML = `
        <table>
            <thead><tr><th>Metrică</th><th>kNN-VC</th><th>YourTTS</th><th>FreeVC</th><th>Voice Clone</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

// Export CSV
document.getElementById('export-compare-btn')?.addEventListener('click', () => {
    if (!lastCompareResult) return;

    const knn = lastCompareResult.knn_vc;
    const yourtts = lastCompareResult.yourtts;
    const fvc = lastCompareResult.freevc;
    const clone = lastCompareResult.voice_clone;
    const metrics = ['mcd', 'pesq', 'speaker_similarity', 'f0_rmse', 'f0_pcc', 'snr'];

    let csv = 'Metrica,kNN-VC,YourTTS,FreeVC,VoiceClone\n';
    metrics.forEach(m => {
        const kVal = knn?.metrics?.[m] ?? '';
        const yVal = yourtts?.metrics?.[m] ?? '';
        const fVal = fvc?.metrics?.[m] ?? '';
        const cVal = clone?.metrics?.[m] ?? '';
        csv += `${m},${kVal},${yVal},${fVal},${cVal}\n`;
    });
    csv += `\nTimp inferenta (ms),${knn?.conversion_time_ms ?? ''},${lvc?.conversion_time_ms ?? ''},${fvc?.conversion_time_ms ?? ''},${clone?.conversion_time_ms ?? ''}\n`;
    csv += `Vorbitor tinta,${lastCompareResult.target_speaker},${lastCompareResult.target_speaker},${lastCompareResult.target_speaker},${lastCompareResult.target_speaker}\n`;
    csv += `Timestamp,${lastCompareResult.timestamp},${lastCompareResult.timestamp},${lastCompareResult.timestamp},${lastCompareResult.timestamp}\n`;

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `compare_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast('CSV exportat!', 'success');
});

// ============================================================
// HISTORY
// ============================================================

async function loadHistory() {
    try {
        const data = await apiFetch('/api/history');
        const history = data.history || [];
        const container = document.getElementById('history-list');

        if (!history.length) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon">📁</div><p>Nu există conversii în istoric.</p></div>';
            return;
        }

        container.innerHTML = history.map(h => {
            // Normalize data structure between different models
            const modelName = h.model === 'knn-vc' ? 'kNN-VC' :
                h.model === 'freevc' ? 'FreeVC' :
                    h.model === 'rvc' ? 'RVC' :
                        h.model === 'yourtts' ? 'YourTTS' :
                            h.model === 'VoiceClone-TTS' ? 'Voice Clone' :
                                h.model === 'lightvc' ? 'LightVC' : h.model;

            const modelClass = h.model === 'knn-vc' ? 'knn' :
                h.model === 'freevc' ? 'freevc' :
                    h.model === 'rvc' ? 'rvc' :
                        h.model === 'yourtts' ? 'yourtts' :
                            h.model === 'VoiceClone-TTS' ? 'clone' : 'lightvc';

            const style = h.model === 'freevc' ? 'background-color:#3b82f6;color:white;' :
                h.model === 'rvc' ? 'background-color:#f97316;color:white;' :
                    h.model === 'yourtts' ? 'background-color:#10b981;color:white;' :
                        h.model === 'VoiceClone-TTS' ? 'background-color:#a855f7;color:white;' : '';

            const srcFile = h.source_filename || (h.source_audio ? h.source_audio.split('/').pop() : 'Necunoscut');
            const timeMs = h.conversion_time_ms || (h.stt_time ? h.stt_time * 1000 : 0);
            const timeText = timeMs > 0 ? formatTime(timeMs) : '—';

            // Handle output path for audio source
            let audioSrc = '';
            if (h.output_path) {
                audioSrc = '/api/audio/' + h.output_path.split(/[\\/]/).pop();
            } else if (h.converted_audio) {
                audioSrc = h.converted_audio;
            }

            return `
            <div class="history-item">
                <span class="history-model-badge">
                    <span class="model-tag ${modelClass}" style="${style}">
                        ${modelName}
                    </span>
                </span>
                <div class="history-info">
                    <div class="history-speaker">→ ${h.target_speaker || 'N/A'}</div>
                    <div class="history-file">${srcFile}</div>
                    ${h.source_text ? `<div style="font-size:0.8rem;color:#94a3b8;margin-top:2px;font-style:italic">"${h.source_text}"</div>` : ''}
                </div>
                <span class="history-time-badge">${timeText}</span>
                <span class="history-time-badge">${new Date(h.timestamp).toLocaleTimeString('ro-RO')}</span>
                <div class="history-audio">
                    <audio controls style="height:32px;width:200px">
                        <source src="${audioSrc}" type="audio/wav">
                    </audio>
                </div>
            </div>
            `;
        }).join('');
    } catch (e) {
        console.error('History error:', e);
    }
}

// ============================================================
// VOICE RECORDER MODULE
// ============================================================

/**
 * Convertește un Blob audio (WebM, MP3, etc.) la WAV 16kHz mono pe client.
 * Folosește Web Audio API — nu necesită FFmpeg pe server.
 */
async function convertBlobToWav(inputBlob) {
    const arrayBuffer = await inputBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Resample la 16kHz mono
    const targetSR = 16000;
    const numChannels = 1;
    const offlineCtx = new OfflineAudioContext(numChannels, audioBuffer.duration * targetSR, targetSR);
    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start();

    const renderedBuffer = await offlineCtx.startRendering();
    const channelData = renderedBuffer.getChannelData(0);

    // Encode WAV
    const wavBuffer = encodeWAV(channelData, targetSR);
    audioCtx.close();
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);          // PCM
    view.setUint16(22, 1, true);          // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);         // 16-bit
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Float32 → Int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
}

const recorderState = {
    knn: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    lvc: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    cmp: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    clone: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    freevc: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    rvc: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    yourtts: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    mknn: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null },
    spkrec: { stream: null, recorder: null, chunks: [], interval: null, seconds: 0, audioCtx: null, analyser: null, animFrame: null }
};

// Tab switching
document.querySelectorAll('.source-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const target = tab.dataset.target;
        const mode = tab.dataset.mode;

        // Update tab active state
        document.querySelectorAll(`.source-tab[data-target="${target}"]`).forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Show/hide zones
        const uploadZoneIds = {
            'knn': 'source-upload-zone',
            'lvc': 'lvc-source-upload-zone',
            'freevc': 'freevc-source-upload-zone',
            'rvc': 'rvc-source-upload-zone',
            'yourtts': 'yourtts-source-upload-zone',
            'clone': 'clone-source-upload-zone',
            'cmp': 'cmp-upload-zone',
            'mknn': 'mknn-source-upload-zone',
            'spkrec': 'spkrec-source-upload-zone'
        };
        const recorderIds = {
            'knn': 'knn-recorder',
            'lvc': 'lvc-recorder',
            'freevc': 'freevc-recorder',
            'rvc': 'rvc-recorder',
            'yourtts': 'yourtts-recorder',
            'clone': 'clone-recorder',
            'cmp': 'cmp-recorder',
            'mknn': 'mknn-recorder',
            'spkrec': 'spkrec-recorder'
        };

        const uploadZone = document.getElementById(uploadZoneIds[target]);
        const recorder = document.getElementById(recorderIds[target]);

        if (mode === 'upload') {
            if (uploadZone) uploadZone.classList.remove('hidden');
            if (recorder) recorder.classList.add('hidden');
        } else {
            if (uploadZone) uploadZone.classList.add('hidden');
            if (recorder) recorder.classList.remove('hidden');
        }
    });
});

// Record button handlers
document.querySelectorAll('.btn-record').forEach(btn => {
    btn.addEventListener('click', () => startRecording(btn.dataset.target));
});

document.querySelectorAll('.btn-record-stop').forEach(btn => {
    btn.addEventListener('click', () => stopRecording(btn.dataset.target));
});

async function startRecording(target) {
    const state = recorderState[target];
    const recordBtn = document.getElementById(`${target}-record-btn`);
    const stopBtn = document.getElementById(`${target}-stop-btn`);
    const timer = document.getElementById(`${target}-recorder-timer`);
    const widget = document.getElementById(`${target}-recorder`);
    const canvas = document.getElementById(`${target}-recorder-canvas`);

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
    } catch (err) {
        toast('Nu pot accesa microfonul. Verifică permisiunile browserului.', 'error');
        console.error('Mic error:', err);
        return;
    }

    // Setup analyser for waveform
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const sourceNode = state.audioCtx.createMediaStreamSource(state.stream);
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = 256;
    sourceNode.connect(state.analyser);

    // Start waveform visualization
    drawWaveform(target, canvas);

    // Setup MediaRecorder
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

    state.recorder = new MediaRecorder(state.stream, { mimeType });
    state.chunks = [];

    state.recorder.ondataavailable = e => {
        if (e.data.size > 0) state.chunks.push(e.data);
    };

    state.recorder.onstop = () => {
        handleRecordingComplete(target);
    };

    state.recorder.start(100); // collect data every 100ms

    // Timer
    state.seconds = 0;
    timer.textContent = '00:00';
    timer.classList.add('recording-active');
    state.interval = setInterval(() => {
        state.seconds++;
        const m = Math.floor(state.seconds / 60).toString().padStart(2, '0');
        const s = (state.seconds % 60).toString().padStart(2, '0');
        timer.textContent = `${m}:${s}`;
    }, 1000);

    // UI state
    recordBtn.classList.add('recording');
    widget.classList.add('recording');
    stopBtn.classList.remove('hidden');
}

function stopRecording(target) {
    const state = recorderState[target];
    const recordBtn = document.getElementById(`${target}-record-btn`);
    const stopBtn = document.getElementById(`${target}-stop-btn`);
    const timer = document.getElementById(`${target}-recorder-timer`);
    const widget = document.getElementById(`${target}-recorder`);

    if (state.recorder && state.recorder.state !== 'inactive') {
        state.recorder.stop();
    }

    if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
    }

    if (state.interval) {
        clearInterval(state.interval);
        state.interval = null;
    }

    if (state.animFrame) {
        cancelAnimationFrame(state.animFrame);
        state.animFrame = null;
    }

    if (state.audioCtx) {
        state.audioCtx.close();
        state.audioCtx = null;
    }

    recordBtn.classList.remove('recording');
    widget.classList.remove('recording');
    stopBtn.classList.add('hidden');
    timer.classList.remove('recording-active');
}

async function handleRecordingComplete(target) {
    const state = recorderState[target];
    if (!state.chunks.length) return;

    const webmBlob = new Blob(state.chunks, { type: 'audio/webm' });
    const duration = state.seconds;

    toast(`Înregistrare: ${duration}s — se convertește la WAV...`, 'info');

    // Convert WebM → WAV on client side (no FFmpeg needed on server)
    let blob;
    try {
        blob = await convertBlobToWav(webmBlob);
    } catch (e) {
        console.warn('Client WAV conversion failed, sending WebM:', e);
        blob = webmBlob;
    }

    const localUrl = URL.createObjectURL(blob);

    const previewIds = {
        'knn': { container: 'source-preview', audio: 'source-audio', filename: 'source-filename' },
        'lvc': { container: 'lvc-source-preview', audio: 'lvc-source-audio', filename: 'lvc-source-filename' },
        'freevc': { container: 'freevc-source-preview', audio: 'freevc-source-audio', filename: 'freevc-source-filename' },
        'rvc': { container: 'rvc-source-preview', audio: 'rvc-source-audio', filename: 'rvc-source-filename' },
        'yourtts': { container: 'yourtts-source-preview', audio: 'yourtts-source-audio', filename: 'yourtts-source-filename' },
        'clone': { container: 'clone-source-preview', audio: 'clone-source-audio', filename: 'clone-source-filename' },
        'cmp': { container: 'cmp-source-preview', audio: 'cmp-source-audio', filename: 'cmp-source-filename' },
        'mknn': { container: 'mknn-source-preview', audio: 'mknn-source-audio', filename: 'mknn-source-filename' },
        'spkrec': { container: 'spkrec-source-preview', audio: 'spkrec-source-audio', filename: 'spkrec-source-filename' }
    };

    const ids = previewIds[target];
    const audioEl = document.getElementById(ids.audio);
    if (audioEl) audioEl.src = localUrl;
    if (ids.filename) {
        const fnEl = document.getElementById(ids.filename);
        if (fnEl) fnEl.textContent = `🎙 Înregistrare (${duration}s)`;
    }
    if (ids.container) show(ids.container);

    toast(`Înregistrare: ${duration}s — se procesează...`, 'info');

    // Upload to backend for WAV conversion
    try {
        const fd = new FormData();
        fd.append('audio', blob, `recording_${Date.now()}.wav`);
        fd.append('purpose', 'source');

        const res = await apiFetch('/api/upload-recording', { method: 'POST', body: fd });

        if (res.success) {
            // Create a File-like object for the conversion flows
            const wavResp = await fetch(res.recording_url);
            const wavBlob = await wavResp.blob();
            const wavFile = new File([wavBlob], res.filename || 'recording.wav', { type: 'audio/wav' });

            // Set the source file for the appropriate converter
            if (target === 'knn') {
                sourceFile = wavFile;
                updateConvertBtn();
            } else if (target === 'lvc') {
                lvcSourceFile = wavFile;
                updateLvcConvertBtn();
            } else if (target === 'cmp') {
                cmpSourceFile = wavFile;
                updateCompareBtn();
            } else if (target === 'clone') {
                cloneSourceFile = wavFile;
                updateCloneBtn();
            } else if (target === 'freevc') {
                freevcSourceFile = wavFile;
                updateFreevcConvertBtn();
            } else if (target === 'rvc') {
                rvcSourceFile = wavFile;
                updateRvcConvertBtn();
            } else if (target === 'yourtts') {
                yourttsSourceFile = wavFile;
                updateYourttsConvertBtn();
            } else if (target === 'mknn') {
                mknnSourceFile = wavFile;
                updateMknnConvertBtn();
            } else if (target === 'spkrec') {
                spkrecSourceFile = wavFile;
                document.getElementById('spkrec-recognize-btn').disabled = false;
            }

            // Update audio preview with WAV
            if (audioEl) audioEl.src = res.recording_url;

            toast(`Înregistrare convertită la WAV (${res.duration?.toFixed(1) || duration}s)`, 'success');
        }
    } catch (e) {
        console.error('Upload recording error:', e);
        // Fallback: use the WebM blob directly
        const file = new File([blob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
        if (target === 'knn') { sourceFile = file; updateConvertBtn(); }
        else if (target === 'lvc') { lvcSourceFile = file; updateLvcConvertBtn(); }
        else if (target === 'cmp') { cmpSourceFile = file; updateCompareBtn(); }
        else if (target === 'clone') { cloneSourceFile = file; updateCloneBtn(); }
        else if (target === 'freevc') { freevcSourceFile = file; updateFreevcConvertBtn(); }
        else if (target === 'rvc') { rvcSourceFile = file; updateRvcConvertBtn(); }
        else if (target === 'yourtts') { yourttsSourceFile = file; updateYourttsConvertBtn(); }
        else if (target === 'mknn') { mknnSourceFile = file; updateMknnConvertBtn(); }
        else if (target === 'spkrec') {
            spkrecSourceFile = file;
            document.getElementById('spkrec-recognize-btn').disabled = false;
        }
        toast('Înregistrare salvată (fără conversie WAV)', 'warning');
    }
}

function drawWaveform(target, canvas) {
    const state = recorderState[target];
    if (!state.analyser || !canvas) return;

    const ctx = canvas.getContext('2d');
    const bufferLength = state.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!state.analyser) return;
        state.animFrame = requestAnimationFrame(draw);

        state.analyser.getByteTimeDomainData(dataArray);

        const w = canvas.width = canvas.offsetWidth;
        const h = canvas.height = canvas.offsetHeight;

        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fillRect(0, 0, w, h);

        ctx.lineWidth = 2;
        ctx.strokeStyle = state.recorder?.state === 'recording'
            ? '#ef4444'
            : 'rgba(239,68,68,0.4)';

        ctx.beginPath();
        const sliceWidth = w / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * h) / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Draw center line
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();
    }

    draw();
}

// ============================================================
// VOICE CLONING (STT -> TTS)
// ============================================================

let cloneSourceFile = null;
const cloneZone = document.getElementById('clone-source-upload-zone');
const cloneFileInput = document.getElementById('clone-file-input');

if (cloneZone) {
    cloneZone.addEventListener('click', () => cloneFileInput.click());
    cloneZone.addEventListener('dragover', e => { e.preventDefault(); cloneZone.classList.add('drag-over'); });
    cloneZone.addEventListener('dragleave', () => cloneZone.classList.remove('drag-over'));
    cloneZone.addEventListener('drop', e => {
        e.preventDefault();
        cloneZone.classList.remove('drag-over');
        const file = [...e.dataTransfer.files].find(f => f.type.startsWith('audio/'));
        if (file) handleCloneFile(file);
    });

    cloneFileInput.addEventListener('change', () => {
        if (cloneFileInput.files.length) handleCloneFile(cloneFileInput.files[0]);
    });
}

function handleCloneFile(file) {
    cloneSourceFile = file;
    const txt = document.querySelector('#clone-source-upload-zone .upload-text');
    if (txt) txt.textContent = `📄 ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    const icon = document.querySelector('#clone-source-upload-zone .upload-icon');
    if (icon) icon.textContent = '✅';
    updateCloneBtn();
}

function updateCloneBtn() {
    const spk = document.getElementById('clone-target-speaker-select')?.value;
    const btn = document.getElementById('btn-clone-voice');
    if (btn) btn.disabled = !(cloneSourceFile && spk);
}

document.getElementById('clone-target-speaker-select')?.addEventListener('change', updateCloneBtn);

document.getElementById('btn-clone-voice')?.addEventListener('click', async () => {
    if (!cloneSourceFile) { toast('Alege un fișier audio sursă!', 'error'); return; }

    const spk = document.getElementById('clone-target-speaker-select').value;
    if (!spk) { toast('Alege vorbitorul țintă!', 'error'); return; }

    const loading = document.getElementById('clone-loading');
    const resultBody = document.getElementById('clone-result-body');
    const btn = document.getElementById('btn-clone-voice');

    loading.classList.remove('hidden');
    resultBody.innerHTML = '';
    btn.disabled = true;

    try {
        const fd = new FormData();
        fd.append('source', cloneSourceFile);
        fd.append('speaker_id', spk);

        const res = await apiFetch('/api/tts/clone', { method: 'POST', body: fd });

        loading.classList.add('hidden');

        resultBody.innerHTML = `
            <div class="result-success-banner">✅ Conversie STT → TTS reușită!</div>
            
            <div style="background:var(--bg-lighter);padding:1rem;border-radius:12px;margin:1rem 0;">
                <h4 style="margin-bottom:0.5rem;font-size:0.9rem;color:var(--text-muted)">📝 Text Transcris (Whisper STT):</h4>
                <p style="font-size:1.1rem;font-family:var(--font-mono);line-height:1.5;">"${res.source_text}"</p>
                <div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.5rem">Timp transcriere: ${(res.stt_time || 0).toFixed(1)}s</div>
            </div>

            <div class="result-players">
                <div class="player-box">
                    <h4>Audio Sursă</h4>
                    <audio controls src="${res.source_url}"></audio>
                </div>
                <div class="player-arrow">→</div>
                <div class="player-box highlight">
                    <h4>Voce Nouă (SpeechT5)</h4>
                    <audio controls src="${res.output_url}" class="final-audio"></audio>
                </div>
            </div>
            <div style="margin-top:1rem;text-align:right">
                <a href="${res.output_url}" download="voice_clone_${spk}.wav" class="btn btn-secondary">💾 Descarcă Audio</a>
            </div>
        `;
        toast('Voice Cloning realizat cu succes!', 'success');
        loadHistory();
    } catch (e) {
        loading.classList.add('hidden');
        toast(`Eroare: ${e.message}`, 'error');
        resultBody.innerHTML = `<div class="empty-state"><div class="empty-icon" style="color:var(--red)">❌</div><p>Eroare la procesare.</p><p style="font-size:0.9rem;color:var(--text-muted)">${e.message}</p></div>`;
    } finally {
        updateCloneBtn();
    }
});


// ============================================================
// INIT
// ============================================================

async function init() {
    await loadSystemInfo();
    await loadSpeakersIntoSelects();
    startTrainingPolling();
}

init();
// ============================================================
// CONVERSIE mKNN-VC (XLS-R Pro)
// ============================================================
let mknnSourceFile = null;

function updateMknnConvertBtn() {
    const btn = document.getElementById('mknn-convert-btn');
    const select = document.getElementById('mknn-target-speaker-select');
    if (btn && select) btn.disabled = !mknnSourceFile || !select.value;
}

setupUploadZone('mknn-source-upload-zone', 'mknn-source-file-input', 'mknn-source-preview',
    'mknn-source-filename', 'mknn-source-audio',
    file => { mknnSourceFile = file; updateMknnConvertBtn(); }
);

document.getElementById('mknn-target-speaker-select')?.addEventListener('change', updateMknnConvertBtn);

document.getElementById('mknn-convert-btn')?.addEventListener('click', async () => {
    const speakerId = document.getElementById('mknn-target-speaker-select')?.value;
    const topk = parseInt(document.getElementById('mknn-topk-slider').value);
    if (!mknnSourceFile || !speakerId) return;

    showLoading('mknn-convert-loading');
    hide('mknn-result-card');

    try {
        const fd = new FormData();
        fd.append('source', mknnSourceFile);
        fd.append('target_speaker', speakerId);
        fd.append('topk', topk);

        const res = await apiFetch('/api/convert/mknn', { method: 'POST', body: fd });

        document.getElementById('mknn-result-original').src = res.source_url || res.source;
        document.getElementById('mknn-result-converted').src = res.output_url;
        document.getElementById('mknn-result-info').innerHTML = `
            <span class="result-badge">XLS-R (Pro)</span>
            <span class="result-badge" style="background:rgba(34,197,94,0.15);color:var(--green);border-color:rgba(34,197,94,0.3)">${formatTime(res.conversion_time_ms)}</span>
            <span class="result-badge" style="background:rgba(168,85,247,0.15);color:#d8b4fe;border-color:rgba(168,85,247,0.3)">→ ${speakerId}</span>
            <span class="result-badge" style="background:rgba(255,255,255,0.05);color:var(--text-muted);border-color:rgba(255,255,255,0.05)">k=${topk}</span>
        `;

        show('mknn-result-card');
        toast('Conversie XLS-R finalizata!', 'success');
        loadSystemInfo();
    } catch (e) {
        toast('Eroare conversie: ' + e.message, 'error');
    } finally {
        hideLoading('mknn-convert-loading');
    }
});

// ============================================================
// FREEVC (SpeechT5) FINE-TUNING & GRAPH
// ============================================================

let lossChartFreevc = null;
let freevcTrainingPollInterval = null;

async function refreshFreeVCTrainingStatus() {
    try {
        const status = await apiFetch('/api/freevc/status');
        const isTraining = status.is_training;
        const progress = status.training_progress;

        if (isTraining) {
            document.getElementById('start-freevc-train-btn').disabled = true;
            document.getElementById('start-freevc-train-btn').innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Se antrenează...';
            show('training-freevc-progress-card');
            updateFreeVCProgressUI(progress);
        } else {
            document.getElementById('start-freevc-train-btn').disabled = false;
            document.getElementById('start-freevc-train-btn').innerHTML = '<span class="btn-icon">⚡</span><span>Pornește Fine-Tuning FreeVC</span>';
            if (progress?.status === 'done' || progress?.status === 'finished') {
                show('training-freevc-progress-card');
                updateFreeVCProgressUI(progress);
            }
        }
    } catch (e) {
        console.error('FreeVC Training status error:', e);
    }
}

function updateFreeVCProgressUI(progress) {
    if (!progress) return;
    const epoch = progress.epoch || 0;
    const total = progress.total_epochs || 50;
    const pct = Math.round((epoch / total) * 100);

    document.getElementById('prog-freevc-epoch-text').textContent = `Epoca ${epoch} / ${total}`;
    document.getElementById('prog-freevc-loss-text').textContent = progress.avg_loss ? `Loss: ${progress.avg_loss.toFixed(4)}` : 'Loss: —';
    document.getElementById('prog-freevc-eta-text').textContent = progress.elapsed_hours ? `Timp: ${progress.elapsed_hours.toFixed(2)}h` : '';
    document.getElementById('training-freevc-progress-bar').style.width = `${pct}%`;
    document.getElementById('prog-freevc-message').textContent = progress.message || `Antrenare FreeVC: Epoch ${epoch}`;

    // Update chart
    if (progress.avg_loss) {
        updateLossChartFreeVC(epoch, progress.avg_loss);
    }
}

let freevcEpochsHistory = [];
let freevcLossHistory = [];

function updateLossChartFreeVC(epoch, loss) {
    const ctx = document.getElementById('loss-chart-freevc');
    if (!ctx) return;

    if (!freevcEpochsHistory.includes(epoch)) {
        freevcEpochsHistory.push(epoch);
        freevcLossHistory.push(loss);
    }

    if (lossChartFreevc) {
        lossChartFreevc.data.labels = freevcEpochsHistory;
        lossChartFreevc.data.datasets[0].data = freevcLossHistory;
        lossChartFreevc.update('none');
        return;
    }

    lossChartFreevc = new Chart(ctx, {
        type: 'line',
        data: {
            labels: freevcEpochsHistory,
            datasets: [{
                label: 'Training Loss (FreeVC)',
                data: freevcLossHistory,
                borderColor: 'rgba(59,130,246,0.8)',
                backgroundColor: 'rgba(59,130,246,0.1)',
                borderWidth: 2,
                pointRadius: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { x: { grid: { color: 'rgba(255,255,255,0.04)' } }, y: { grid: { color: 'rgba(255,255,255,0.04)' } } }
        }
    });
}

function startFreeVCTrainingPolling() {
    if (freevcTrainingPollInterval) clearInterval(freevcTrainingPollInterval);
    freevcTrainingPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (!page.classList.contains('active')) return;
        await refreshFreeVCTrainingStatus();
    }, 3000);
}

document.getElementById('start-freevc-train-btn')?.addEventListener('click', async () => {
    if (!confirm('Pornești antrenarea (Fine-Tuning) pentru FreeVC?\nAcest proces necesită GPU AMD și VRAM liber.')) return;
    try {
        const btn = document.getElementById('start-freevc-train-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Pornire...';

        const res = await apiFetch('/api/freevc/train', { method: 'POST' });
        toast(res.message || 'Fine-Tuning FreeVC pornit!', 'success');

        show('training-freevc-progress-card');
        await refreshFreeVCTrainingStatus();
        startFreeVCTrainingPolling();
    } catch (err) {
        toast(err.message, 'error');
        const btn = document.getElementById('start-freevc-train-btn');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span><span>Pornește Fine-Tuning FreeVC</span>';
    }
});

// Pornim polling-ul la inițializare
startFreeVCTrainingPolling();

// Adauga suport pentru fișiere înregistrate
const origHandle = handleRecordingComplete;
handleRecordingComplete = async function (target) {
    await origHandle(target);
    if (target === 'mknn' && window.lastRecordedWavFile) {
        mknnSourceFile = window.lastRecordedWavFile;
        updateMknnConvertBtn();
    }
}

// =====================================================================
// AFISARE GRAFICE ON-DEMAND (SPECTROGRAME)
// =====================================================================
function initSpectrogramButtons() {
    document.querySelectorAll('.toggle-plots-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const target = this.getAttribute('data-target');
            const plotsContainer = document.getElementById(`cmp-${target}-plots`);

            if (!plotsContainer) return;

            // Toggle visibility
            if (!plotsContainer.classList.contains('hidden') && plotsContainer.style.display !== 'none' && plotsContainer.style.display !== '') {
                plotsContainer.classList.add('hidden');
                plotsContainer.style.display = 'none';
                return;
            }

            plotsContainer.classList.remove('hidden');
            plotsContainer.style.display = 'block';

            // Check if already loaded
            const specImg = document.getElementById(`cmp-${target}-plot-spectrogram`);
            if (specImg && specImg.src && specImg.src.includes('/api/plot')) {
                return; // Deja incarcat in memorie
            }

            // Aratam mesajul loading
            const loading = document.getElementById(`cmp-${target}-plot-loading`);
            if (loading) loading.style.display = 'block';

            // Gasim numele fisierului generat din URL
            let filename = null;
            let audioPlayer = null;

            if (target === 'source') {
                audioPlayer = document.getElementById('cmp-source-audio');
            } else {
                audioPlayer = document.getElementById(`cmp-${target}-audio`);
            }

            if (audioPlayer && audioPlayer.src) {
                filename = audioPlayer.src.split('/').pop();
                // eliminam posibili parametri de query
                if (filename.includes('?')) {
                    filename = filename.split('?')[0];
                }
            }

            if (!filename || filename === 'null' || filename === '') {
                if (loading) loading.innerText = '⚠️ Trebuie sa efectuezi conversia mai intai pentru a vedea graficele!';
                return;
            }

            // Trimiterea cererii on-demand catre server
            specImg.src = `/api/plot?filename=${filename}&plot_type=spectrogram`;
            const melImg = document.getElementById(`cmp-${target}-plot-mel`);
            melImg.src = `/api/plot?filename=${filename}&plot_type=mel`;
            const mfccImg = document.getElementById(`cmp-${target}-plot-mfcc`);
            mfccImg.src = `/api/plot?filename=${filename}&plot_type=mfcc`;

            // La incarcare, ascundem loading-ul
            specImg.onload = () => {
                if (loading) loading.style.display = 'none';
                specImg.style.display = 'inline-block';
                melImg.style.display = 'inline-block';
                mfccImg.style.display = 'inline-block';
            };

            // Logica eroare (daca nu exista)
            specImg.onerror = () => {
                if (loading) loading.innerText = '⚠️ Eroare la generarea graficelor din RAM.';
            };
        });
    });
}

// Apelăm funcția direct (deoarece scriptul e încărcat la finalul DOM-ului)
initSpectrogramButtons();

// ============================================================
// XTTS FINE-TUNING & RVC TRAINING
// ============================================================

let xttsPollInterval = null;
let rvcPollInterval = null;

async function refreshXTTSStatus() {
    try {
        const progress = await apiFetch('/api/xtts/finetune/status');
        const btn = document.getElementById('start-xtts-train-btn');

        if (progress.status === 'training' || progress.status === 'cleaning' || progress.status === 'preparing') {
            btn.disabled = true;
            btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Se procesează...';
            show('training-xtts-progress-card');

            const pct = progress.progress_pct || 0;
            document.getElementById('prog-xtts-epoch-text').textContent = progress.epoch ? `Epoca ${progress.epoch} / ${progress.total_epochs || '?'}` : 'Pregătire date...';
            document.getElementById('prog-xtts-loss-text').textContent = progress.loss ? `Loss: ${progress.loss.toFixed(4)}` : '';
            document.getElementById('prog-xtts-eta-text').textContent = progress.elapsed_hours ? `Timp: ${progress.elapsed_hours.toFixed(2)}h` : '';
            document.getElementById('training-xtts-progress-bar').style.width = `${pct}%`;
            document.getElementById('prog-xtts-message').textContent = progress.message || 'Se procesează...';
        } else {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">⚡</span><span>Pornește Fine-Tuning XTTS pe Română</span>';
            }
            if (progress.status === 'done' || progress.status === 'error') {
                show('training-xtts-progress-card');
                document.getElementById('training-xtts-progress-bar').style.width = '100%';
                document.getElementById('prog-xtts-message').textContent = progress.message || 'Finalizat.';
            }
        }
    } catch (e) {
        console.error('XTTS Training status error:', e);
    }
}

async function refreshRVCStatus() {
    try {
        const progress = await apiFetch('/api/rvc/status');
        const btn = document.getElementById('start-rvc-train-btn');

        if (progress.status === 'training' || progress.status === 'preparing') {
            btn.disabled = true;
            btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Se antrenează...';
            show('training-rvc-progress-card');

            const pct = progress.progress_pct || 0;
            document.getElementById('training-rvc-progress-bar').style.width = `${pct}%`;
            document.getElementById('prog-rvc-message').textContent = progress.message || 'Se antrenează RVC...';
        } else {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">🎭</span><span>Antrenează Model RVC</span>';
            }
            if (progress.status === 'done' || progress.status === 'error') {
                show('training-rvc-progress-card');
                document.getElementById('training-rvc-progress-bar').style.width = '100%';
                document.getElementById('prog-rvc-message').textContent = progress.message || 'Finalizat.';
            }
        }
    } catch (e) {
        console.error('RVC Training status error:', e);
    }
}

function startXTTSAndRVCPolling() {
    if (xttsPollInterval) clearInterval(xttsPollInterval);
    if (rvcPollInterval) clearInterval(rvcPollInterval);

    xttsPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (page && page.classList.contains('active')) await refreshXTTSStatus();
    }, 3000);

    rvcPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (page && page.classList.contains('active')) await refreshRVCStatus();
    }, 3000);
}

document.getElementById('start-xtts-train-btn')?.addEventListener('click', async () => {
    if (!confirm('Pornești fine-tuning-ul XTTS pe dataset-ul Common Voice RO?\nPe AMD GPU, antrenarea se va face pe CPU și va dura câteva ore.')) return;
    try {
        const btn = document.getElementById('start-xtts-train-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Pornire...';

        const res = await apiFetch('/api/xtts/finetune', { method: 'POST' });
        toast(res.message || 'Fine-Tuning XTTS pornit!', 'success');

        show('training-xtts-progress-card');
        await refreshXTTSStatus();
    } catch (err) {
        toast(err.message, 'error');
        const btn = document.getElementById('start-xtts-train-btn');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">⚡</span><span>Pornește Fine-Tuning XTTS pe Română</span>';
    }
});

document.getElementById('start-rvc-train-btn')?.addEventListener('click', async () => {
    const spk = document.getElementById('rvc-speaker-select').value;
    if (!spk) {
        toast('Te rog selectează un vorbitor țintă mai întâi!', 'error');
        return;
    }

    if (!confirm('Pornești antrenarea modelului RVC pentru acest vorbitor?')) return;

    try {
        const btn = document.getElementById('start-rvc-train-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Pornire...';

        const fd = new FormData();
        fd.append('speaker_id', spk);

        const res = await apiFetch('/api/rvc/train', { method: 'POST', body: fd });
        toast(res.message || 'Antrenare RVC pornită!', 'success');

        show('training-rvc-progress-card');
        await refreshRVCStatus();
    } catch (err) {
        toast(err.message, 'error');
        const btn = document.getElementById('start-rvc-train-btn');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🎭</span><span>Antrenează Model RVC</span>';
    }
});

// ============================================================
// ANTRENARE YourTTS
// ============================================================

let yourttsPollInterval = null;
let yourttsChart = null;

async function refreshYourTTSStatus() {
    try {
        const data = await apiFetch('/api/yourtts/status');
        const training = data.training || {};

        if (training.status === 'preparing' || training.status === 'training') {
            show('training-yourtts-progress-card');
            const pct = training.progress_pct || 0;
            document.getElementById('training-yourtts-progress-bar').style.width = pct + '%';
            document.getElementById('prog-yourtts-message').textContent =
                training.message || `Epoca ${training.epoch}/${training.total_epochs}`;

            // Desenare grafic
            if (training.loss_history && training.loss_history.length > 0) {
                const ctx = document.getElementById('yourttsTrainingChart');
                if (ctx) {
                    if (!yourttsChart) {
                        yourttsChart = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: Array.from({ length: training.loss_history.length }, (_, i) => i + 1),
                                datasets: [{
                                    label: 'Loss',
                                    data: training.loss_history,
                                    borderColor: '#10b981',
                                    tension: 0.1,
                                    fill: true,
                                    backgroundColor: 'rgba(16, 185, 129, 0.1)'
                                }]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                animation: false,
                                scales: { y: { beginAtZero: false } }
                            }
                        });
                    } else {
                        yourttsChart.data.labels = Array.from({ length: training.loss_history.length }, (_, i) => i + 1);
                        yourttsChart.data.datasets[0].data = training.loss_history;
                        yourttsChart.update();
                    }
                }
            }

            const btn = document.getElementById('start-yourtts-train-btn');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Antrenare...';
            }
        } else {
            const btn = document.getElementById('start-yourtts-train-btn');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-icon">🌍</span><span>Pornește Fine-Tuning YourTTS</span>';
            }
            if (training.status === 'done' || training.status === 'error') {
                show('training-yourtts-progress-card');
                document.getElementById('training-yourtts-progress-bar').style.width = '100%';
                document.getElementById('prog-yourtts-message').textContent = training.message || 'Finalizat.';
            }
        }
    } catch (e) {
        console.error('YourTTS Training status error:', e);
    }
}

document.getElementById('start-yourtts-train-btn')?.addEventListener('click', async () => {
    if (!confirm('Pornești fine-tuning-ul YourTTS pe dataset-ul Common Voice RO?\nAceastă operațiune va dura 4-8 ore pe CPU.')) return;
    try {
        const btn = document.getElementById('start-yourtts-train-btn');
        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner" style="width:16px;height:16px;display:inline-block"></span> Pornire...';

        const res = await apiFetch('/api/yourtts/train', { method: 'POST' });
        toast(res.message || 'Fine-Tuning YourTTS pornit!', 'success');

        show('training-yourtts-progress-card');
        await refreshYourTTSStatus();
    } catch (err) {
        toast(err.message, 'error');
        const btn = document.getElementById('start-yourtts-train-btn');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🌍</span><span>Pornește Fine-Tuning YourTTS</span>';
    }
});

// Pornim polling-ul si pentru XTTS, RVC și YourTTS
function startAllTrainingPolling() {
    if (xttsPollInterval) clearInterval(xttsPollInterval);
    if (rvcPollInterval) clearInterval(rvcPollInterval);
    if (yourttsPollInterval) clearInterval(yourttsPollInterval);

    xttsPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (page && page.classList.contains('active')) await refreshXTTSStatus();
    }, 3000);

    rvcPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (page && page.classList.contains('active')) await refreshRVCStatus();
    }, 3000);

    yourttsPollInterval = setInterval(async () => {
        const page = document.getElementById('page-training');
        if (page && page.classList.contains('active')) await refreshYourTTSStatus();
    }, 3000);
}

startAllTrainingPolling();


// ============================================================
// CONVERSIE RVC
// ============================================================

let rvcSourceFile = null;

setupUploadZone('rvc-source-upload-zone', 'rvc-source-file-input', 'rvc-source-preview',
    'rvc-source-filename', 'rvc-source-audio',
    (file) => { rvcSourceFile = file; checkRvcConvertBtn(); }
);

function checkRvcConvertBtn() {
    const hasAudio = rvcSourceFile;
    const hasTarget = document.getElementById('rvc-target-speaker-select')?.value;
    const btn = document.getElementById('rvc-convert-btn');
    if (btn) btn.disabled = !(hasAudio && hasTarget);
}

document.getElementById('rvc-target-speaker-select')?.addEventListener('change', checkRvcConvertBtn);

document.getElementById('rvc-convert-btn')?.addEventListener('click', async () => {
    const targetSpk = document.getElementById('rvc-target-speaker-select').value;
    const audioBlob = rvcSourceFile;
    if (!targetSpk || !audioBlob) return;

    const formData = new FormData();
    formData.append('source', audioBlob, rvcSourceFile ? rvcSourceFile.name : 'record.webm');
    formData.append('speaker_id', targetSpk);

    const loading = document.getElementById('rvc-convert-loading');
    const resultCard = document.getElementById('rvc-result-card');

    loading.classList.remove('hidden');
    resultCard.classList.add('hidden');

    try {
        const res = await apiFetch('/api/rvc/convert', {
            method: 'POST',
            body: formData,
            headers: {}
        });

        document.getElementById('rvc-result-original').src = res.source_url;
        document.getElementById('rvc-result-converted').src = res.output_url;

        document.getElementById('rvc-original-stt').textContent = res.source_text ? `"${res.source_text}"` : '';
        document.getElementById('rvc-converted-stt').textContent = res.converted_text ? `"${res.converted_text}"` : '';

        let infoHtml = `<strong>Timp procesare:</strong> ${res.conversion_time_ms.toFixed(0)} ms | <strong>Durată:</strong> ${res.duration.toFixed(2)} s`;
        if (res.wer !== null) {
            infoHtml += ` | <strong>WER:</strong> ${(res.wer * 100).toFixed(1)}%`;
        }
        document.getElementById('rvc-result-info').innerHTML = infoHtml;
        resultCard.classList.remove('hidden');

        loadHistory();
        loadSystemInfo();
        toast('Conversie RVC completă!', 'success');
    } catch (e) {
        toast('Eroare conversie RVC: ' + e.message, 'error');
    } finally {
        loading.classList.add('hidden');
    }
});

const navConvertRvc = document.getElementById('nav-convert-rvc');
if (navConvertRvc) {
    navConvertRvc.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        navConvertRvc.classList.add('active');
        document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
        document.getElementById('page-convert-rvc').classList.add('active');
    });
}

// ============================================================
// CONVERSIE YourTTS
// ============================================================

let yourttsSourceFile = null;

setupUploadZone('yourtts-source-upload-zone', 'yourtts-source-file-input', 'yourtts-source-preview',
    'yourtts-source-filename', 'yourtts-source-audio',
    (file) => { yourttsSourceFile = file; checkYourttsConvertBtn(); }
);

function checkYourttsConvertBtn() {
    const hasAudio = yourttsSourceFile;
    const hasTarget = document.getElementById('yourtts-target-speaker-select')?.value;
    const btn = document.getElementById('yourtts-convert-btn');
    if (btn) btn.disabled = !(hasAudio && hasTarget);
}

document.getElementById('yourtts-target-speaker-select')?.addEventListener('change', checkYourttsConvertBtn);

document.getElementById('yourtts-convert-btn')?.addEventListener('click', async () => {
    const targetSpk = document.getElementById('yourtts-target-speaker-select').value;
    const audioBlob = yourttsSourceFile;
    if (!targetSpk || !audioBlob) return;

    const formData = new FormData();
    formData.append('source', audioBlob, yourttsSourceFile ? yourttsSourceFile.name : 'record.webm');
    formData.append('speaker_id', targetSpk);

    const loading = document.getElementById('yourtts-convert-loading');
    const resultCard = document.getElementById('yourtts-result-card');

    loading.classList.remove('hidden');
    resultCard.classList.add('hidden');

    try {
        const res = await apiFetch('/api/yourtts/convert', {
            method: 'POST',
            body: formData,
            headers: {}
        });

        document.getElementById('yourtts-result-original').src = res.source_url;
        document.getElementById('yourtts-result-converted').src = res.output_url;

        //document.getElementById('yourtts-original-stt').textContent = res.source_text ? `"${res.source_text}"` : '';
        // document.getElementById('yourtts-converted-stt').textContent = res.converted_text ? `"${res.converted_text}"` : '';

        let infoHtml = `<strong>Timp procesare:</strong> ${res.conversion_time_ms.toFixed(0)} ms | <strong>Durată:</strong> ${res.duration.toFixed(2)} s`;
        if (res.wer !== null && res.wer !== undefined) {
            infoHtml += ` `;
        }
        document.getElementById('yourtts-result-info').innerHTML = infoHtml;
        resultCard.classList.remove('hidden');

        loadHistory();
        loadSystemInfo();
        toast('Conversie YourTTS completă!', 'success');
    } catch (e) {
        toast('Eroare conversie YourTTS: ' + e.message, 'error');
    } finally {
        loading.classList.add('hidden');
    }
});

const navConvertYourtts = document.getElementById('nav-convert-yourtts');
if (navConvertYourtts) {
    navConvertYourtts.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        navConvertYourtts.classList.add('active');
        document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
        document.getElementById('page-convert-yourtts').classList.add('active');
    });
}

// ============================================================
// RECUNOAȘTERE VORBITOR (Speaker Recognition)
// ============================================================

let spkrecSourceFile = null;

// Upload zone click & drag
(function () {
    const zone = document.getElementById('spkrec-source-upload-zone');
    const input = document.getElementById('spkrec-file-input');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) handleSpkrecFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => {
        if (input.files[0]) handleSpkrecFile(input.files[0]);
    });
})();

function handleSpkrecFile(file) {
    spkrecSourceFile = file;
    const preview = document.getElementById('spkrec-source-preview');
    const audio = document.getElementById('spkrec-source-audio');
    const filename = document.getElementById('spkrec-source-filename');
    if (audio) audio.src = URL.createObjectURL(file);
    if (filename) filename.textContent = file.name;
    if (preview) preview.classList.remove('hidden');
    document.getElementById('spkrec-recognize-btn').disabled = false;
    // Hide old results
    document.getElementById('spkrec-results')?.classList.add('hidden');
}

// Recognize button handler
document.getElementById('spkrec-recognize-btn')?.addEventListener('click', async () => {
    if (!spkrecSourceFile) { toast('Încarcă sau înregistrează un audio!', 'warning'); return; }

    const loading = document.getElementById('spkrec-loading');
    const resultsDiv = document.getElementById('spkrec-results');
    const resultsList = document.getElementById('spkrec-results-list');
    const btn = document.getElementById('spkrec-recognize-btn');

    btn.disabled = true;
    loading?.classList.remove('hidden');
    resultsDiv?.classList.add('hidden');

    try {
        const fd = new FormData();
        fd.append('audio', spkrecSourceFile);

        const res = await apiFetch('/api/speakers/recognize', { method: 'POST', body: fd });

        if (res.success && res.all_results.length > 0) {
            resultsList.innerHTML = res.all_results.map((r, i) => {
                const barColor = r.is_match
                    ? (i === 0 ? '#10b981' : '#3b82f6')
                    : '#6b7280';
                const matchBadge = r.is_match
                    ? '<span style="background:#10b981;color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;margin-left:8px;">\u2713 Match</span>'
                    : '';
                const bestBadge = i === 0 && r.is_match
                    ? '<span style="background:linear-gradient(135deg,#a855f7,#7c3aed);color:white;padding:2px 10px;border-radius:12px;font-size:0.75rem;margin-left:8px;">\uD83C\uDFC6 Best Match</span>'
                    : '';
                return `
                <div style="padding:12px 16px;margin-bottom:8px;background:var(--card-bg, #1e293b);border-radius:10px;border:1px solid ${r.is_match ? barColor : 'var(--border, #334155)'};">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                        <div style="display:flex;align-items:center;">
                            <span style="font-weight:600;font-size:1.05rem;">${r.name}</span>
                            ${bestBadge}${matchBadge}
                        </div>
                        <span style="font-weight:700;font-size:1.1rem;color:${barColor};">${r.score.toFixed(1)}%</span>
                    </div>
                    <div style="background:var(--bg, #0f172a);border-radius:6px;height:8px;overflow:hidden;">
                        <div style="height:100%;width:${r.score}%;background:${barColor};border-radius:6px;transition:width 0.6s ease;"></div>
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-muted, #94a3b8);margin-top:4px;">
                        Similaritate cosinus: ${r.raw_similarity} | Referințe: ${r.num_references}
                    </div>
                </div>`;
            }).join('');
            resultsDiv?.classList.remove('hidden');
            toast('Recunoaștere finalizată!', 'success');
        } else {
            resultsList.innerHTML = '<div class="empty-state"><p>Nu s-au putut calcula rezultate. Verifică dacă vorbitorii au fișiere de referință.</p></div>';
            resultsDiv?.classList.remove('hidden');
        }
    } catch (e) {
        toast('Eroare recunoaștere: ' + e.message, 'error');
    } finally {
        loading?.classList.add('hidden');
        btn.disabled = false;
    }
});
