const api = {
    start: () => fetch('/api/start').then(r => r.ok),
    stop: () => fetch('/api/stop').then(r => r.ok),
    status: () => fetch('/api/status').then(r => r.json())
};

const dom = {
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    connection: document.getElementById('connection-status'),
    connText: document.getElementById('conn-text'),
    submissionCount: document.getElementById('submission-count'),
    budgetVal: document.getElementById('budget-val'),
    budgetFill: document.getElementById('budget-fill'),
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

// Start Polling
setInterval(updateStatus, 1000);
updateStatus();
