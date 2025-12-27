const api = {
    start: () => fetch('/api/start').then(r => r.ok),
    stop: () => fetch('/api/stop').then(r => r.ok),
    status: () => fetch('/api/status').then(r => r.json()),
    genKey: () => fetch('/api/generate_key').then(r => r.json())
};

const dom = {
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    connection: document.getElementById('connection-status'),
    connText: document.getElementById('conn-text'),
    submissionCount: document.getElementById('submission-count'),
    budgetVal: document.getElementById('budget-val'),
    budgetFill: document.getElementById('budget-fill'),
    savedMb: document.getElementById('saved-mb'),
    latTrain: document.getElementById('lat-train'),
    latCompress: document.getElementById('lat-compress'),
    latEncrypt: document.getElementById('lat-encrypt'),
    compRatio: document.getElementById('comp-ratio'),
    mseVal: document.getElementById('mse-val'),
    auditLog: document.getElementById('audit-log'),
    keyPath: document.getElementById('key-path'),
    keyBadge: document.getElementById('key-badge'),
    btnGenKey: document.getElementById('btn-gen-key'),
    genStatus: document.getElementById('gen-status'),
    pipeline: document.querySelector('.pipeline'),
    simdBadge: document.getElementById('simd-badge'),
    weights: {
        visual: document.getElementById('weight-visual'),
        language: document.getElementById('weight-language'),
        auxiliary: document.getElementById('weight-aux')
    },
    experts: {
        visual: document.getElementById('expert-visual'),
        language: document.getElementById('expert-language'),
        auxiliary: document.getElementById('expert-aux')
    }
};

let isRunning = false;

// Poll Status
async function updateStatus() {
    try {
        const data = await api.status();

        // Connection
        dom.connection.className = `status-badge ${data.connection === 'connected' ? 'connected' : 'disconnected'}`;
        dom.connText.innerText = data.connection === 'connected' ? 'Secure Link' : 'Offline';

        // Stats
        dom.submissionCount.innerText = data.submissions;
        dom.budgetVal.innerText = data.privacy_budget;
        dom.budgetFill.style.width = `${data.budget_percent}%`;

        // Key Info
        dom.keyPath.innerText = data.key_path;
        if (data.key_exists) {
            dom.keyBadge.innerText = "LOCKED (READY)";
            dom.keyBadge.className = "badge locked";
        } else {
            dom.keyBadge.innerText = "MISSING";
            dom.keyBadge.className = "badge missing";
        }

        // Telemetry
        if (data.telemetry) {
            dom.savedMb.innerText = `${data.telemetry.bandwidth_saved_mb.toFixed(1)} MB`;
            dom.latTrain.innerText = `${data.telemetry.latency_train.toFixed(1)}ms`;
            dom.latCompress.innerText = `${data.telemetry.latency_compress.toFixed(1)}ms`;
            dom.latEncrypt.innerText = `${data.telemetry.latency_encrypt.toFixed(1)}ms`;
            dom.compRatio.innerText = `${data.telemetry.compression_ratio.toFixed(0)}:1`;
            dom.mseVal.innerText = data.telemetry.quality_mse.toFixed(6);
        }

        // Audit Log
        if (data.audit && data.audit.length > 0) {
            dom.auditLog.innerHTML = data.audit.reverse().map(entry => `
                <div class="audit-entry">
                    <span class="time">${entry.timestamp.split('T')[1].split('.')[0]}</span>
                    <span class="event">${entry.event}</span>
                    <span class="key">${entry.key_id}</span>
                </div>
            `).join('');
        }

        // SIMD
        if (data.simd) dom.simdBadge.classList.remove('hidden');
        else dom.simdBadge.classList.add('hidden');

        // Experts
        if (data.experts) {
            for (const [key, weight] of Object.entries(data.experts)) {
                if (dom.weights[key]) dom.weights[key].innerText = `${weight}x`;
                if (dom.experts[key]) {
                    if (isRunning) dom.experts[key].classList.add('active');
                    else dom.experts[key].classList.remove('active');
                }
            }
        }

        // State Sync
        if (data.running !== isRunning) {
            isRunning = data.running;
            updateControls();
        }

    } catch (e) {
        console.error("Status fetch failed", e);
        dom.connection.className = 'status-badge disconnected';
        dom.connText.innerText = 'Server Error';
    }
}

function updateControls() {
    dom.btnStart.disabled = isRunning;
    dom.btnStop.disabled = !isRunning;

    if (isRunning) {
        dom.pipeline.classList.add('active');
    } else {
        dom.pipeline.classList.remove('active');
    }
}

// Listeners
dom.btnStart.onclick = async () => {
    await api.start();
    updateStatus();
};

dom.btnStop.onclick = async () => {
    await api.stop();
    updateStatus();
};

dom.btnGenKey.onclick = async () => {
    dom.btnGenKey.disabled = true;
    dom.genStatus.innerText = "Generating 128-bit N2HE key...";
    try {
        const res = await api.genKey();
        if (res.status === 'success') {
            dom.genStatus.innerText = `Success: ${res.path}`;
            dom.genStatus.className = "status-msg success";
        } else {
            throw new Error(res.message || "Key generation failed");
        }
    } catch (e) {
        dom.genStatus.innerText = `Error: ${e.message}`;
        dom.genStatus.className = "status-msg error";
    } finally {
        dom.btnGenKey.disabled = false;
        setTimeout(() => { dom.genStatus.innerText = ""; }, 5000);
        updateStatus();
    }
};

// Start Polling
setInterval(updateStatus, 1000);
updateStatus();
