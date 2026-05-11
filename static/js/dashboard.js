/**
 * FactoryGuard AI - Dashboard JavaScript
 * Handles real-time charts, API calls, and UI interactions
 * Includes Prediction History & Analytics
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── Global State ──────────────────────────────────────────────
    let dashboardData = null;
    let charts = {};
    let simulationTimer = null;
    let isSimulating = false;

    // ── DOM Elements ──────────────────────────────────────────────
    const ui = {
        navItems: document.querySelectorAll('.nav-item'),
        sections: document.querySelectorAll('.content-section'),
        sectionTitle: document.getElementById('sectionTitle'),
        currentTime: document.getElementById('currentTime'),
        menuToggle: document.getElementById('menuToggle'),
        sidebar: document.getElementById('sidebar'),

        // Overview
        kpiPrecision: document.getElementById('kpi-precision'),
        kpiRecall: document.getElementById('kpi-recall'),
        kpiF1: document.getElementById('kpi-f1'),
        kpiPrauc: document.getElementById('kpi-prauc'),

        // Live Sensors
        simBtn: document.getElementById('startSimulation'),
        simStatus: document.getElementById('sim-status'),
        predLog: document.getElementById('predictionLog'),
        clearLogBtn: document.getElementById('clearLog'),

        // Prediction Form
        predictBtn: document.getElementById('predictBtn'),
        predOutput: document.getElementById('predictionOutput'),
        predStatus: document.getElementById('pred-status'),
        predProb: document.getElementById('pred-prob'),
        predRisk: document.getElementById('pred-risk'),
        predAction: document.getElementById('pred-action'),

        // Model Info
        modelName: document.getElementById('model-name'),
        modelFeatures: document.getElementById('model-features'),
        modelAcc: document.getElementById('model-accuracy'),
        cmTN: document.getElementById('cm-tn'),
        cmFP: document.getElementById('cm-fp'),
        cmFN: document.getElementById('cm-fn'),
        cmTP: document.getElementById('cm-tp'),

        // Sensor Cards
        valAirTemp: document.getElementById('val-air-temp'),
        valProcessTemp: document.getElementById('val-process-temp'),
        valRpm: document.getElementById('val-rpm'),
        valTorque: document.getElementById('val-torque'),
        valToolWear: document.getElementById('val-tool-wear'),
        barAirTemp: document.getElementById('bar-air-temp'),
        barProcessTemp: document.getElementById('bar-process-temp'),
        barRpm: document.getElementById('bar-rpm'),
        barTorque: document.getElementById('bar-torque'),
        barToolWear: document.getElementById('bar-tool-wear'),
        predResult: document.getElementById('prediction-result'),
        predProbCard: document.getElementById('prediction-prob'),
        riskBadge: document.getElementById('risk-badge'),
        consensusGrid: document.getElementById('consensus-grid'),

        // ✅ History Elements
        totalPredictions: document.getElementById('totalPredictions'),
        criticalCount: document.getElementById('criticalCount'),
        highCount: document.getElementById('highCount'),
        mediumCount: document.getElementById('mediumCount'),
        lowCount: document.getElementById('lowCount'),
        predictionTableBody: document.getElementById('predictionTableBody')
    };

    // ── Chart Configurations ──────────────────────────────────────
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Inter';

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#f1f5f9', font: { size: 12 } }
            }
        },
        scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, border: { display: false } },
            y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, border: { display: false } }
        }
    };

    // ── Initialization ────────────────────────────────────────────
    init();

    function init() {
        updateClock();
        setInterval(updateClock, 1000);
        setupNavigation();
        initCharts();
        fetchDashboardData();
        setupEventListeners();
        
        // ✅ Initialize History
        loadPredictionHistory();
        loadPredictionStats();
        
        // Auto-refresh history every 30 seconds
        setInterval(() => {
            loadPredictionHistory();
            loadPredictionStats();
        }, 30000);
    }

    // ── Utility Functions ─────────────────────────────────────────
    function updateClock() {
        const now = new Date();
        ui.currentTime.textContent = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    // ── Navigation ────────────────────────────────────────────────
    function setupNavigation() {
        ui.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = item.getAttribute('data-section');
                
                // Update active nav
                ui.navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');
                
                // Update title
                ui.sectionTitle.textContent = item.querySelector('span').textContent;

                // Show target section
                ui.sections.forEach(sec => sec.classList.remove('active'));
                document.getElementById(`section-${targetId}`).classList.add('active');

                // ✅ Refresh history when history tab is clicked
                if (targetId === 'history') {
                    loadPredictionHistory();
                    loadPredictionStats();
                }

                // Close mobile menu
                if (window.innerWidth <= 768) {
                    ui.sidebar.classList.remove('open');
                }
            });
        });

        ui.menuToggle.addEventListener('click', () => {
            ui.sidebar.classList.toggle('open');
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ✅ PREDICTION HISTORY & ANALYTICS - NEW FUNCTIONALITY
    // ═══════════════════════════════════════════════════════════════════════

    async function loadPredictionHistory() {
        try {
            const response = await fetch('/api/prediction-history?limit=50');
            const data = await response.json();
            
            if (data.success) {
                displayPredictionHistory(data.predictions);
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    async function loadPredictionStats() {
        try {
            const response = await fetch('/api/prediction-statistics');
            const data = await response.json();
            
            if (data.success) {
                const stats = data.statistics;
                if (ui.totalPredictions) ui.totalPredictions.textContent = stats.total_predictions || 0;
                if (ui.criticalCount) ui.criticalCount.textContent = stats.risk_distribution.CRITICAL || 0;
                if (ui.highCount) ui.highCount.textContent = stats.risk_distribution.HIGH || 0;
                if (ui.mediumCount) ui.mediumCount.textContent = stats.risk_distribution.MEDIUM || 0;
                if (ui.lowCount) ui.lowCount.textContent = stats.risk_distribution.LOW || 0;
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    function displayPredictionHistory(predictions) {
        if (!ui.predictionTableBody) return;
        
        ui.predictionTableBody.innerHTML = '';
        
        if (predictions.length === 0) {
            ui.predictionTableBody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px;">No predictions yet</td></tr>';
            return;
        }
        
        predictions.reverse().forEach(pred => {
            const timestamp = new Date(pred.timestamp).toLocaleString();
            const predictionText = pred.prediction === 1 ? '⚠️ FAILURE' : '✅ NORMAL';
            
            let riskClass = 'risk-low';
            if (pred.risk_level === 'CRITICAL') riskClass = 'risk-critical';
            else if (pred.risk_level === 'HIGH') riskClass = 'risk-high';
            else if (pred.risk_level === 'MEDIUM') riskClass = 'risk-medium';
            
            const row = `
                <tr>
                    <td>${timestamp}</td>
                    <td>${predictionText}</td>
                    <td>${(pred.failure_probability * 100).toFixed(2)}%</td>
                    <td><span class="${riskClass}">${pred.risk_level}</span></td>
                    <td>${parseFloat(pred.air_temp).toFixed(1)}°C</td>
                    <td>${parseFloat(pred.process_temp).toFixed(1)}°C</td>
                    <td>${parseFloat(pred.rotational_speed).toFixed(0)}</td>
                    <td>${parseFloat(pred.torque).toFixed(2)}</td>
                    <td>${parseFloat(pred.tool_wear).toFixed(1)}</td>
                    <td>${(pred.ensemble_agreement * 100).toFixed(1)}%</td>
                </tr>
            `;
            ui.predictionTableBody.innerHTML += row;
        });
    }

    async function downloadReport() {
        try {
            const response = await fetch('/api/export-report');
            const data = await response.json();
            alert('✅ Report exported successfully!\nLocation: ' + data.file);
        } catch (error) {
            console.error('Error downloading report:', error);
            alert('❌ Error downloading report');
        }
    }

    // Make functions globally available for inline onclick
    window.loadPredictionHistory = loadPredictionHistory;
    window.loadPredictionStats = loadPredictionStats;
    window.downloadReport = downloadReport;

    // ── Data Fetching & UI Update ─────────────────────────────────
    async function fetchDashboardData() {
        try {
            const res = await fetch('/api/dashboard-data');
            dashboardData = await res.json();
            updateDashboardUI(dashboardData);
        } catch (error) {
            console.error("Error fetching dashboard data:", error);
        }
    }

    function updateDashboardUI(data) {
        // Overview KPIs
        ui.kpiPrecision.textContent = (data.metrics.precision * 100).toFixed(1) + '%';
        ui.kpiRecall.textContent = (data.metrics.recall * 100).toFixed(1) + '%';
        ui.kpiF1.textContent = (data.metrics.f1 * 100).toFixed(1) + '%';
        ui.kpiPrauc.textContent = data.metrics.pr_auc.toFixed(4);

        // Model Info
        ui.modelName.textContent = data.metrics.model_name;
        ui.modelFeatures.textContent = '15 Core Features'; // Based on config
        ui.modelAcc.textContent = (data.metrics.accuracy * 100).toFixed(1) + '%';

        // Confusion Matrix
        const cm = data.confusion_matrix;
        ui.cmTN.textContent = cm[0][0];
        ui.cmFP.textContent = cm[0][1];
        ui.cmFN.textContent = cm[1][0];
        ui.cmTP.textContent = cm[1][1];

        // Update Charts
        updateCharts(data);
    }

    // ── Chart Rendering ───────────────────────────────────────────
    function initCharts() {
        // Sensor Telemetry Chart
        const ctxSensor = document.getElementById('sensorChart').getContext('2d');
        charts.sensor = new Chart(ctxSensor, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...commonOptions,
                elements: { point: { radius: 0 }, line: { tension: 0.4, borderWidth: 2 } },
                interaction: { mode: 'index', intersect: false }
            }
        });

        // Failure Distribution Chart
        const ctxFailure = document.getElementById('failureTypeChart').getContext('2d');
        charts.failure = new Chart(ctxFailure, {
            type: 'doughnut',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#f1f5f9', padding: 20 } }
                }
            }
        });

        // Feature Importance Chart
        const ctxImportance = document.getElementById('featureImportanceChart').getContext('2d');
        charts.importance = new Chart(ctxImportance, {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: {
                ...commonOptions,
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: { x: { display: false }, y: { grid: { display: false } } }
            }
        });

        // Risk Distribution Chart
        const ctxRisk = document.getElementById('riskChart').getContext('2d');
        charts.risk = new Chart(ctxRisk, {
            type: 'doughnut',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: { legend: { position: 'right', labels: { color: '#f1f5f9' } } }
            }
        });

        // PR Curve Chart
        const ctxPR = document.getElementById('prCurveChart').getContext('2d');
        charts.prCurve = new Chart(ctxPR, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...commonOptions,
                elements: { point: { radius: 0 }, line: { tension: 0.1, borderWidth: 3 } },
                scales: {
                    x: { type: 'linear', title: { display: true, text: 'Recall', color: '#94a3b8' }, min: 0, max: 1 },
                    y: { title: { display: true, text: 'Precision', color: '#94a3b8' }, min: 0, max: 1.05 }
                }
            }
        });

        // Live Feed Chart (Sensors Section)
        const ctxLive = document.getElementById('liveChart').getContext('2d');
        charts.live = new Chart(ctxLive, {
            type: 'line',
            data: {
                labels: Array.from({length: 30}, (_, i) => i),
                datasets: [
                    { label: 'Torque', borderColor: '#8b5cf6', data: [], fill: true, backgroundColor: 'rgba(139, 92, 246, 0.1)' }
                ]
            },
            options: {
                ...commonOptions,
                animation: false,
                elements: { point: { radius: 0 }, line: { tension: 0.4, borderWidth: 2 } },
                scales: { x: { display: false } }
            }
        });
    }

    function updateCharts(data) {
        // 1. Sensor Telemetry
        const sensorLen = data.sensor_data.torque.length;
        const labels = Array.from({length: sensorLen}, (_, i) => i);
        
        charts.sensor.data = {
            labels: labels,
            datasets: [
                { label: 'Torque (Nm)', borderColor: '#8b5cf6', data: data.sensor_data.torque },
                { label: 'Air Temp (K)', borderColor: '#3b82f6', data: data.sensor_data.air_temp },
                { label: 'Tool Wear', borderColor: '#f59e0b', data: data.sensor_data.tool_wear }
            ]
        };
        charts.sensor.update();

        // 2. Failure Distribution
        const failTypes = data.failure_distribution.failure_types;
        charts.failure.data = {
            labels: ['Normal', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
            datasets: [{
                data: [
                    data.failure_distribution.total_normal,
                    failTypes.TWF || 0,
                    failTypes.HDF || 0,
                    failTypes.PWF || 0,
                    failTypes.OSF || 0,
                    failTypes.RNF || 0
                ],
                backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899'],
                borderWidth: 0
            }]
        };
        charts.failure.update();

        // 3. Feature Importance
        const ftLabels = data.feature_importance.map(f => f.feature);
        const ftVals = data.feature_importance.map(f => f.mean_abs_shap);
        
        charts.importance.data = {
            labels: ftLabels,
            datasets: [{
                data: ftVals,
                backgroundColor: '#3b82f6',
                borderRadius: 4
            }]
        };
        charts.importance.update();

        // 4. Risk Distribution
        const risk = data.risk_distribution;
        charts.risk.data = {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{
                data: [risk.LOW, risk.MEDIUM, risk.HIGH, risk.CRITICAL],
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#b91c1c'],
                borderWidth: 0
            }]
        };
        charts.risk.update();

        // 5. PR Curve
        charts.prCurve.data = {
            labels: data.pr_curve.recall,
            datasets: [{
                label: `LightGBM (AUC: ${data.metrics.pr_auc.toFixed(4)})`,
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                fill: true,
                data: data.pr_curve.precision
            }]
        };
        charts.prCurve.update();
    }

    // ── Live Simulation ──────────────────────────────────────���────
    function setupEventListeners() {
        ui.simBtn.addEventListener('click', toggleSimulation);
        ui.clearLogBtn.addEventListener('click', clearLog);
        ui.predictBtn.addEventListener('click', handleManualPrediction);
    }

    function toggleSimulation() {
        if (isSimulating) {
            clearInterval(simulationTimer);
            isSimulating = false;
            ui.simBtn.innerHTML = '<i class="fas fa-play"></i> Start Simulation';
            ui.simBtn.style.background = 'var(--gradient-blue)';
            ui.simStatus.textContent = 'Idle';
            ui.simStatus.style.background = 'rgba(255,255,255,0.1)';
            ui.simStatus.style.color = '#94a3b8';
        } else {
            isSimulating = true;
            ui.simBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Simulation';
            ui.simBtn.style.background = 'var(--gradient-red, linear-gradient(135deg, #ef4444, #b91c1c))';
            ui.simStatus.textContent = 'Simulating...';
            ui.simStatus.style.background = 'rgba(16, 185, 129, 0.2)';
            ui.simStatus.style.color = '#10b981';
            
            // Clear empty log message
            if (ui.predLog.querySelector('.log-empty')) {
                ui.predLog.innerHTML = '';
            }
            
            simulateTick();
            simulationTimer = setInterval(simulateTick, 2000); // Fetch every 2 seconds
        }
    }

    async function simulateTick() {
        try {
            const res = await fetch('/api/simulate');
            const data = await res.json();
            
            updateSensorCards(data);
            updateLiveChart(data);
            addLogEntry(data);
            
            // ✅ Auto-refresh history on each simulation tick
            loadPredictionStats();
        } catch (error) {
            console.error("Simulation error:", error);
        }
    }

    function updateSensorCards(data) {
        const sensors = data.sensor_readings;
        
        ui.valAirTemp.textContent = sensors.air_temp;
        ui.valProcessTemp.textContent = sensors.process_temp;
        ui.valRpm.textContent = sensors.rotational_speed;
        ui.valTorque.textContent = sensors.torque;
        ui.valToolWear.textContent = sensors.tool_wear;

        // Update progress bars (conceptual normalization)
        ui.barAirTemp.style.width = Math.min((sensors.air_temp - 295) / 10 * 100, 100) + '%';
        ui.barProcessTemp.style.width = Math.min((sensors.process_temp - 305) / 10 * 100, 100) + '%';
        ui.barRpm.style.width = Math.min(sensors.rotational_speed / 3000 * 100, 100) + '%';
        ui.barTorque.style.width = Math.min(sensors.torque / 80 * 100, 100) + '%';
        ui.barToolWear.style.width = Math.min(sensors.tool_wear / 250 * 100, 100) + '%';

        // Update Prediction Card
        const isFailure = data.prediction === 1;
        ui.predResult.textContent = isFailure ? 'FAILURE IMMINENT' : 'NORMAL';
        ui.predResult.style.color = isFailure ? '#ef4444' : '#10b981';
        
        ui.predProbCard.textContent = `Failure Probability: ${(data.failure_probability * 100).toFixed(1)}%`;
        
        ui.riskBadge.className = `risk-badge risk-${data.risk_level}`;
        ui.riskBadge.textContent = `${data.risk_level} RISK`;
    }

    function updateLiveChart(data) {
        const limit = 30;
        const torqueData = charts.live.data.datasets[0].data;
        
        torqueData.push(data.sensor_readings.torque);
        if (torqueData.length > limit) torqueData.shift();
        
        charts.live.update();
    }

    function addLogEntry(data) {
        const isFailure = data.prediction === 1;
        const probability = (data.failure_probability * 100).toFixed(1);
        
        let modelsInfo = "";
        if (data.ensemble_predictions) {
            Object.keys(data.ensemble_predictions).forEach(k => {
                const isFail = data.ensemble_predictions[k].prediction === 1;
                modelsInfo += `[${k.replace(' ', '')}: ${isFail ? 'FAIL' : 'NORM'}] `;
            });
        }
        
        const entry = document.createElement('div');
        entry.className = `log-entry ${isFailure ? 'failure' : 'normal'}`;
        
        entry.innerHTML = `
            <div class="log-time">${data.timestamp.split(' ')[1]}</div>
            <div class="log-status" style="color: ${isFailure ? 'var(--accent-red)' : 'var(--accent-green)'}">
                ${isFailure ? 'FAILURE' : 'NORMAL'}
            </div>
            <div class="log-details" style="display: flex; flex-direction: column; gap: 4px;">
                <div>
                    Risk: <strong>${data.risk_level}</strong> | 
                    Prob: ${probability}% | 
                    Tq: ${data.sensor_readings.torque}Nm | 
                    Wear: ${data.sensor_readings.tool_wear}m
                </div>
                <div style="font-size: 11px; opacity: 0.7;">
                    ${modelsInfo}
                </div>
            </div>
        `;
        
        ui.predLog.prepend(entry);
        
        // Keep max 50 entries
        if (ui.predLog.children.length > 50) {
            ui.predLog.removeChild(ui.predLog.lastChild);
        }
    }

    function clearLog() {
        ui.predLog.innerHTML = '<div class="log-empty">Start simulation to see predictions</div>';
    }

    // ── Manual Prediction ─────────────────────────────────────────
    async function handleManualPrediction() {
        // Gather inputs
        const inputData = {
            "air_temp": parseFloat(document.getElementById('input-air-temp').value),
            "process_temp": parseFloat(document.getElementById('input-process-temp').value),
            "rotational_speed": parseInt(document.getElementById('input-rpm').value),
            "torque": parseFloat(document.getElementById('input-torque').value),
            "tool_wear": parseInt(document.getElementById('input-tool-wear').value),
            // Dummy values for one-hot cols that API handles
            "Type_L": 1, 
            "Type_M": 0
        };

        const btn = ui.predictBtn;
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        btn.disabled = true;

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            const data = await res.json();
            
            if (data.success) {
                showPredictionResult(data);
                // ✅ Refresh history after prediction
                loadPredictionHistory();
                loadPredictionStats();
            } else {
                alert('Prediction failed: ' + data.error);
            }
        } catch (error) {
            console.error("Prediction error:", error);
            alert('Failed to connect to API');
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    function showPredictionResult(data) {
        ui.predOutput.classList.remove('hidden');
        
        const isFailure = data.prediction === 1;
        
        ui.predStatus.textContent = isFailure ? 'FAILURE' : 'NORMAL';
        ui.predStatus.style.color = isFailure ? '#ef4444' : '#10b981';
        
        ui.predProb.textContent = (data.failure_probability * 100).toFixed(2) + '%';
        
        ui.predRisk.textContent = data.risk_level;
        ui.predRisk.className = `pred-value`;
        
        // Color risk level
        if (data.risk_level === 'LOW') ui.predRisk.style.color = '#10b981';
        else if (data.risk_level === 'MEDIUM') ui.predRisk.style.color = '#f59e0b';
        else ui.predRisk.style.color = '#ef4444';
        
        ui.predAction.textContent = isFailure ? 'MAINTENANCE REQUIRED' : 'CONTINUE OPERATION';
        ui.predAction.style.color = isFailure ? '#ef4444' : '#text-secondary';

        // Render consensus
        if (data.ensemble_predictions && ui.consensusGrid) {
            ui.consensusGrid.innerHTML = '';
            for (const [name, info] of Object.entries(data.ensemble_predictions)) {
                const fail = info.prediction === 1;
                const prob = (info.failure_probability * 100).toFixed(1) + '%';
                
                const card = document.createElement('div');
                card.className = 'pred-result-card';
                card.style.padding = '16px';
                card.innerHTML = `
                    <div class="pred-label">${name}</div>
                    <div class="pred-value" style="font-size: 16px; color: ${fail ? '#ef4444' : '#10b981'}">
                        ${fail ? 'FAIL' : 'NORMAL'}
                    </div>
                    <div class="pred-label" style="margin-top: 4px; font-size: 11px;">Prob: ${prob}</div>
                `;
                ui.consensusGrid.appendChild(card);
            }
        }
    }
});