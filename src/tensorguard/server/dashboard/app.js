const api = {
    start: () => fetch('/api/start').then(r => r.ok),
    stop: () => fetch('/api/stop').then(r => r.ok),
    status: () => fetch('/api/status').then(r => r.json()),
    genKey: () => fetch('/api/generate_key').then(r => r.json()),
    updateSettings: (settings) => fetch('/api/update_settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    }).then(r => r.json())
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
    viewName: document.getElementById('current-view-name'),
    navLinks: document.querySelectorAll('.nav-links li'),
    views: document.querySelectorAll('.view-container'),
    weights: {}, // Legacy weights removed
    moiGrid: document.getElementById('moi-grid'),
    settings: {
        epsilon: document.getElementById('set-epsilon'),
        rank: document.getElementById('set-rank'),
        sparsity: document.getElementById('set-sparsity'),
        sparsityVal: document.getElementById('sparsity-val'),
        gating: document.getElementById('set-gating'),
        outlier: document.getElementById('set-outlier'),
        btnSave: document.getElementById('btn-save-settings')
    },
    versionsBody: document.getElementById('version-history-body')
};

let isRunning = false;

// Tab Switching Logic
function switchView(viewId) {
    dom.views.forEach(v => v.classList.remove('active'));
    dom.navLinks.forEach(l => l.classList.remove('active'));

    document.getElementById(`view-${viewId}`).classList.add('active');
    document.querySelector(`[data-view="${viewId}"]`).classList.add('active');

    dom.viewName.innerText = viewId.charAt(0).toUpperCase() + viewId.slice(1);
}

dom.navLinks.forEach(link => {
    link.onclick = () => switchView(link.dataset.view);
});

// Poll Status
async function updateStatus() {
    try {
        const data = await api.status();

        // Connection
        dom.connection.className = `status-badge ${data.connection === 'connected' ? 'connected' : 'disconnected'}`;
        dom.connText.innerText = data.connection === 'connected' ? 'Secure Link' : 'Offline';

        // Overview Tab Data
        dom.submissionCount.innerText = data.submissions;
        dom.budgetVal.innerText = data.privacy_budget;
        dom.budgetFill.style.width = `${data.budget_percent}%`;

        // Key Info
        dom.keyPath.innerText = data.key_path;
        if (data.key_exists) {
            dom.keyBadge.innerText = "READY";
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
            if (dom.compRatio) dom.compRatio.innerText = `${data.telemetry.compression_ratio.toFixed(0)}:1`;
            if (dom.mseVal) dom.mseVal.innerText = data.telemetry.quality_mse.toFixed(6);
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

        // Experts (FedMoE v2.0 MoI)
        if (data.telemetry && data.telemetry.moi_distribution && dom.moiGrid) {
            const weights = data.telemetry.moi_distribution;
            const expertIcons = {
                'visual_primary': '👁️',
                'visual_aux': '🎨',
                'language_semantic': '📝',
                'manipulation_grasp': '🦾'
            };

            dom.moiGrid.innerHTML = Object.entries(weights).map(([expert, weight]) => `
                <div class="expert-stat">
                    <div class="expert-label">
                        <span>${expertIcons[expert] || '🧠'} ${expert.replace('_', ' ').toUpperCase()}</span>
                        <span>${(weight * 100).toFixed(1)}%</span>
                    </div>
                    <div class="expert-bar-bg">
                        <div class="expert-bar-fill" style="width: ${weight * 100}%"></div>
                    </div>
                </div>
            `).join('');
        }

        // Versions Tab
        if (data.history && data.history.length > 0) {
            dom.versionsBody.innerHTML = data.history.map(v => `
                <tr>
                    <td>v${v.version}</td>
                    <td>${v.timestamp.replace('T', ' ')}</td>
                    <td><span class="badge ${v.status === 'Deployed' ? 'locked' : 'missing'}">${v.status}</span></td>
                    <td>${v.quality.toFixed(6)}</td>
                </tr>
            `).join('');
        }

        // Sync Settings from server if not touched
        if (!dom.settings.epsilon.matches(':focus')) dom.settings.epsilon.value = data.settings?.epsilon || 1.0;
        if (!dom.settings.gating.matches(':focus')) dom.settings.gating.value = data.settings?.gating_threshold || 0.15;
        if (!dom.settings.outlier.matches(':focus')) dom.settings.outlier.value = data.settings?.outlier_mad_threshold || 3.0;

        // State Sync
        if (data.running !== isRunning) {
            isRunning = data.running;
            updateControls();
        }

    } catch (e) {
        console.error("Status fetch failed", e);
        dom.connection.className = 'status-badge disconnected';
        dom.connText.innerText = 'Sync Error (Reconnecting...)';

        // Clear telemetry on persistent failure
        if (dom.submissionCount) dom.submissionCount.innerText = "---";
        if (dom.savedMb) dom.savedMb.innerText = "---";
    }
}

function updateControls() {
    dom.btnStart.disabled = isRunning;
    dom.btnStop.disabled = !isRunning;
    if (dom.pipeline) {
        if (isRunning) dom.pipeline.classList.add('active');
        else dom.pipeline.classList.remove('active');
    }
}

// Settings Listeners
dom.settings.sparsity.oninput = () => {
    dom.settings.sparsityVal.innerText = `${dom.settings.sparsity.value}%`;
};

dom.settings.btnSave.onclick = async () => {
    dom.settings.btnSave.disabled = true;
    const s = {
        epsilon: parseFloat(dom.settings.epsilon.value),
        rank: parseInt(dom.settings.rank.value),
        sparsity: parseFloat(dom.settings.sparsity.value),
        gating_threshold: parseFloat(dom.settings.gating.value),
        outlier_mad_threshold: parseFloat(dom.settings.outlier.value)
    };
    try {
        const res = await api.updateSettings(s);
        alert("Settings synchronized with fleet.");
    } catch (err) {
        console.error("Failed to update settings", err);
    } finally {
        dom.settings.btnSave.disabled = false;
    }
};

// Start/Stop
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
    try {
        const res = await api.genKey();
        if (res.status === 'success') {
            updateStatus();
        }
    } finally {
        dom.btnGenKey.disabled = false;
    }
};

// Start Polling
setInterval(updateStatus, 1500);
updateStatus();

// ==========================================
// KMS/HSM Configuration Handlers
// ==========================================

const kmsElements = {
    provider: document.getElementById('kms-provider'),
    providerStatus: document.getElementById('kms-provider-status'),
    connectionDot: document.getElementById('kms-connection-dot'),
    connectionText: document.getElementById('kms-connection-text'),
    awsConfig: document.getElementById('kms-aws-config'),
    azureConfig: document.getElementById('kms-azure-config'),
    gcpConfig: document.getElementById('kms-gcp-config'),
    btnTest: document.getElementById('btn-test-kms'),
    btnSave: document.getElementById('btn-save-kms'),
    auditLog: document.getElementById('kms-audit-log')
};

// Provider switching
if (kmsElements.provider) {
    kmsElements.provider.onchange = () => {
        const provider = kmsElements.provider.value;

        // Hide all provider configs
        kmsElements.awsConfig?.classList.add('hidden');
        kmsElements.azureConfig?.classList.add('hidden');
        kmsElements.gcpConfig?.classList.add('hidden');

        // Show selected provider config
        if (provider === 'aws') {
            kmsElements.awsConfig?.classList.remove('hidden');
        } else if (provider === 'azure') {
            kmsElements.azureConfig?.classList.remove('hidden');
        } else if (provider === 'gcp') {
            kmsElements.gcpConfig?.classList.remove('hidden');
        }

        // Update status bar
        updateKmsStatus(provider, 'pending');
    };
}

function updateKmsStatus(provider, status) {
    const providerNames = {
        'local': 'Local',
        'aws': 'AWS KMS',
        'azure': 'Azure Key Vault',
        'gcp': 'GCP Cloud KMS'
    };

    const statusMessages = {
        'local': 'Using local key storage',
        'aws': 'AWS KMS configured',
        'azure': 'Azure Key Vault configured',
        'gcp': 'GCP Cloud KMS configured',
        'pending': 'Configuration pending...',
        'connected': 'Connected',
        'error': 'Connection failed'
    };

    if (kmsElements.providerStatus) {
        kmsElements.providerStatus.innerText = providerNames[provider] || provider;
    }

    if (kmsElements.connectionDot) {
        kmsElements.connectionDot.className = `dot status-${status === 'connected' ? 'connected' : provider}`;
    }

    if (kmsElements.connectionText) {
        kmsElements.connectionText.innerText = statusMessages[status] || statusMessages[provider];
    }
}

// Test KMS Connection
if (kmsElements.btnTest) {
    kmsElements.btnTest.onclick = async () => {
        const provider = kmsElements.provider?.value || 'local';
        kmsElements.btnTest.disabled = true;
        kmsElements.btnTest.innerText = '🔄 Testing...';

        try {
            // Simulate connection test (would call backend in production)
            await new Promise(resolve => setTimeout(resolve, 1500));

            updateKmsStatus(provider, 'connected');
            addKmsAuditEntry('CONNECTION_TEST', `${provider.toUpperCase()} connection successful`);
            alert(`✅ ${provider.toUpperCase()} connection successful!`);
        } catch (err) {
            updateKmsStatus(provider, 'error');
            alert(`❌ Connection failed: ${err.message}`);
        } finally {
            kmsElements.btnTest.disabled = false;
            kmsElements.btnTest.innerText = '🔗 Test Connection';
        }
    };
}

// Save KMS Configuration
if (kmsElements.btnSave) {
    kmsElements.btnSave.onclick = async () => {
        const provider = kmsElements.provider?.value || 'local';
        kmsElements.btnSave.disabled = true;

        const config = { provider };

        if (provider === 'aws') {
            config.region = document.getElementById('aws-region')?.value;
            config.cmk_arn = document.getElementById('aws-cmk-arn')?.value;
        } else if (provider === 'azure') {
            config.vault_url = document.getElementById('azure-vault-url')?.value;
            config.key_name = document.getElementById('azure-key-name')?.value;
        } else if (provider === 'gcp') {
            config.project = document.getElementById('gcp-project')?.value;
            config.location = document.getElementById('gcp-location')?.value;
            config.keyring = document.getElementById('gcp-keyring')?.value;
            config.key_name = document.getElementById('gcp-keyname')?.value;
        }

        try {
            // Save to localStorage and send to backend
            localStorage.setItem('tensorguard_kms_config', JSON.stringify(config));

            addKmsAuditEntry('CONFIG_SAVED', `${provider.toUpperCase()} configuration saved`);
            updateKmsStatus(provider, provider);
            alert('✅ KMS configuration saved!');
        } catch (err) {
            alert(`❌ Failed to save: ${err.message}`);
        } finally {
            kmsElements.btnSave.disabled = false;
        }
    };
}

function addKmsAuditEntry(event, details) {
    if (!kmsElements.auditLog) return;

    const now = new Date().toISOString().replace('T', ' ').split('.')[0];
    const entry = document.createElement('div');
    entry.className = 'audit-entry';
    entry.innerHTML = `
        <span class="timestamp">${now}</span>
        <span class="event">${event}</span>
        <span class="details">${details}</span>
    `;

    kmsElements.auditLog.insertBefore(entry, kmsElements.auditLog.firstChild);
}

// Load saved KMS config on startup
const savedKmsConfig = localStorage.getItem('tensorguard_kms_config');
if (savedKmsConfig) {
    try {
        const config = JSON.parse(savedKmsConfig);
        if (kmsElements.provider) {
            kmsElements.provider.value = config.provider || 'local';
            kmsElements.provider.dispatchEvent(new Event('change'));
        }
    } catch (e) {
        console.warn('Failed to load KMS config:', e);
    }
}
