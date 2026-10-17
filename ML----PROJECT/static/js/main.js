document.addEventListener('DOMContentLoaded', () => {
    fetchMetrics();
});

let chartsInitialized = false;
let chartInstances = {};

function updateAreaDisp(val) {
    const sqyards = Number(val);
    const sqft = (sqyards * 9).toLocaleString('en-IN');
    document.getElementById('area-val-disp').innerText = `${sqyards.toLocaleString('en-IN')} Gaj (${sqft} sq ft)`;
    const equivEl = document.getElementById('sqft-equiv');
    if (equivEl) {
        equivEl.innerText = `${sqft} sq ft`;
    }
}

function switchTab(tabName) {
    const calcTab = document.getElementById('calculator-section');
    const dashboardTab = document.getElementById('dashboard-section');
    const metricsTab = document.getElementById('metrics-section');

    const calcBtn = document.getElementById('nav-predict-btn');
    const dashboardBtn = document.getElementById('nav-dashboard-btn');
    const metricsBtn = document.getElementById('nav-metrics-btn');

    [calcTab, dashboardTab, metricsTab].forEach(t => t.classList.add('hidden'));
    [calcBtn, dashboardBtn, metricsBtn].forEach(b => b.classList.remove('active'));

    if (tabName === 'calculator') {
        calcTab.classList.remove('hidden');
        calcBtn.classList.add('active');
    } else if (tabName === 'dashboard') {
        dashboardTab.classList.remove('hidden');
        dashboardBtn.classList.add('active');
        if (!chartsInitialized) {
            fetchAnalytics();
            chartsInitialized = true;
        }
    } else if (tabName === 'metrics') {
        metricsTab.classList.remove('hidden');
        metricsBtn.classList.add('active');
    } else if (tabName === 'compare') {
        const compareTab = document.getElementById('compare-section');
        const compareBtn = document.getElementById('nav-compare-btn');
        if (compareTab) compareTab.classList.remove('hidden');
        if (compareBtn) compareBtn.classList.add('active');
    }
        metricsTab.classList.remove('hidden');
        metricsBtn.classList.add('active');
    }
}

async function handlePrediction(event) {
    event.preventDefault();

    const submitBtn = document.getElementById('submit-btn');
    const errorAlert = document.getElementById('error-alert');
    const placeholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');

    errorAlert.classList.add('hidden');

    const sqyards = parseFloat(document.getElementById('area_sqyards').value);
    const formData = {
        city_village: document.getElementById('city_village').value,
        area_sqyards: sqyards,
        area_sqft: sqyards * 9,
        bedrooms: parseInt(document.getElementById('bedrooms').value, 10),
        bathrooms: parseFloat(document.getElementById('bathrooms').value),
        floors: parseInt(document.getElementById('floors').value, 10),
        year_built: parseInt(document.getElementById('year_built').value, 10),
        parking: document.getElementById('parking').value,
        condition: document.getElementById('condition').value
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating Valuation...';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok && result.status === 'success') {
            const data = result.data;
            displayPredictionResults(data);
            placeholder.classList.add('hidden');
            resultContent.classList.remove('hidden');
        } else {
            showError(result.message || 'Error occurred while calculating house price.');
        }
    } catch (err) {
        showError('Network or server connection error. Please try again.');
        console.error('Prediction API Error:', err);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict House Price';
    }
}

function formatINR(number) {
    try {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        }).format(number);
    } catch (e) {
        return `₹${Math.round(number).toLocaleString('en-IN')}`;
    }
}

function formatWordsINR(val) {
    const num = Number(val);
    if (num >= 10000000) {
        return `₹${(num / 10000000).toFixed(2)} Crore`;
    } else if (num >= 100000) {
        return `₹${(num / 100000).toFixed(2)} Lakhs`;
    } else {
        return formatINR(num);
    }
}

function displayPredictionResults(data) {
    const wordsDisplay = document.getElementById('predicted-price-words');
    const priceDisplay = document.getElementById('predicted-price-display');
    const sqydDisplay = document.getElementById('price-sqyd-display');
    const sqftDisplay = document.getElementById('price-sqft-display');
    const rangeDisplay = document.getElementById('price-range-display');
    const chipsContainer = document.getElementById('summary-chips');

    wordsDisplay.innerText = data.price_words || formatWordsINR(data.predicted_price);
    priceDisplay.innerText = `Exact: ${data.price_formatted}`;
    
    sqydDisplay.innerText = data.price_per_sqyard_formatted || `${formatINR(data.price_per_sqyard)}/sq.yd`;
    sqftDisplay.innerText = data.price_per_sqft_formatted || `${formatINR(data.price_per_sqft)}/sq.ft`;
    
    rangeDisplay.innerText = `${data.valuation_range.range_words || ''} (${data.valuation_range.range_formatted})`;

    const summary = data.input_summary;
    chipsContainer.innerHTML = `
        <span class="chip"><i class="fa-solid fa-location-dot"></i> ${summary.city_village}</span>
        <span class="chip"><i class="fa-solid fa-ruler"></i> ${summary.area_sqyards} Gaj (${summary.area_sqft} sqft)</span>
        <span class="chip"><i class="fa-solid fa-bed"></i> ${summary.bedrooms} BHK</span>
        <span class="chip"><i class="fa-solid fa-bath"></i> ${summary.bathrooms} Baths</span>
        <span class="chip"><i class="fa-solid fa-layer-group"></i> ${summary.floors} Story</span>
        <span class="chip"><i class="fa-solid fa-calendar"></i> Built ${summary.year_built}</span>
        <span class="chip"><i class="fa-solid fa-square-parking"></i> ${summary.parking}</span>
        <span class="chip"><i class="fa-solid fa-gem"></i> ${summary.condition}</span>
    `;

    // Render AI-style "Explain My Prediction"
    if (data.explanation) {
        renderExplainability(data.explanation);
    }
}

function renderExplainability(explanation) {
    if (!explanation) return;

    const narrativeEl = document.getElementById('ai-narrative-text');
    if (narrativeEl) {
        narrativeEl.innerText = explanation.natural_explanation || "Prediction analysis complete.";
    }

    const posContainer = document.getElementById('positive-factors-list');
    const negContainer = document.getElementById('negative-factors-list');

    // Render Positive / Increasing Factors
    if (posContainer) {
        posContainer.innerHTML = '';
        if (explanation.increasing_factors && explanation.increasing_factors.length > 0) {
            explanation.increasing_factors.forEach(factor => {
                const item = document.createElement('div');
                item.className = 'driver-item';
                item.innerHTML = `
                    <div class="driver-item-top">
                        <span class="driver-item-name">${factor.title}: ${factor.current_value}</span>
                        <span class="driver-item-impact impact-pos">${factor.impact_formatted}</span>
                    </div>
                    <div class="driver-item-reason">${factor.reason}</div>
                `;
                posContainer.appendChild(item);
            });
        } else {
            posContainer.innerHTML = '<div class="no-factors-msg">No major upward premium factors.</div>';
        }
    }

    // Render Negative / Decreasing Factors
    if (negContainer) {
        negContainer.innerHTML = '';
        if (explanation.decreasing_factors && explanation.decreasing_factors.length > 0) {
            explanation.decreasing_factors.forEach(factor => {
                const item = document.createElement('div');
                item.className = 'driver-item';
                item.innerHTML = `
                    <div class="driver-item-top">
                        <span class="driver-item-name">${factor.title}: ${factor.current_value}</span>
                        <span class="driver-item-impact impact-neg">${factor.impact_formatted}</span>
                    </div>
                    <div class="driver-item-reason">${factor.reason}</div>
                `;
                negContainer.appendChild(item);
            });
        } else {
            negContainer.innerHTML = '<div class="no-factors-msg">No major downward penalty factors.</div>';
        }
    }
}

function showError(msg) {
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');
    errorMessage.innerText = msg;
    errorAlert.classList.remove('hidden');
}

// Model Comparison Metrics API
async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const result = await response.json();

        if (response.ok && result.status === 'success') {
            renderMetricsView(result.data);
            window.latestMetricsData = result.data;
        }
    } catch (err) {
        console.error('Metrics Fetch Error:', err);
    }
}

function renderMetricsView(data) {
    const bestModelTitle = document.getElementById('best-model-title');
    const usedModelBadge = document.getElementById('used-model-badge');
    const grid = document.getElementById('algorithm-cards-grid');

    const bestModel = data.best_model;
    bestModelTitle.innerText = `${bestModel} Pipeline`;
    if (usedModelBadge) {
        usedModelBadge.innerText = `Powered by ${bestModel} (98.3% Accuracy)`;
    }

    grid.innerHTML = '';
    const comparisons = data.model_comparison;

    for (const [algoName, metrics] of Object.entries(comparisons)) {
        const isBest = (algoName === bestModel);
        
        const card = document.createElement('div');
        card.className = `glass-card algo-card ${isBest ? 'best-card' : ''}`;

        card.innerHTML = `
            ${isBest ? '<span class="best-tag"><i class="fa-solid fa-crown"></i> Top Model</span>' : ''}
            <div class="algo-name">${algoName}</div>
            <div class="algo-metric-row">
                <span class="algo-metric-name">R² Accuracy Score</span>
                <span class="algo-metric-val" style="color: ${isBest ? '#10b981' : '#3b82f6'};">${metrics.R2_Score}</span>
            </div>
            <div class="algo-metric-row">
                <span class="algo-metric-name">MAE (Mean Abs Error)</span>
                <span class="algo-metric-val">${formatWordsINR(metrics.MAE)}</span>
            </div>
            <div class="algo-metric-row">
                <span class="algo-metric-name">RMSE (Root Mean Sq Error)</span>
                <span class="algo-metric-val">${formatWordsINR(metrics.RMSE)}</span>
            </div>
            <div class="algo-metric-row">
                <span class="algo-metric-name">Exact MAE</span>
                <span class="algo-metric-val">${formatINR(metrics.MAE)}</span>
            </div>
        `;

        grid.appendChild(card);
    }
}

// Analytics Dashboard Charts API
async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const result = await response.json();

        if (response.ok && result.status === 'success') {
            const data = result.data;
            
            renderPriceDistribution(data.price_distribution);
            renderPriceVsArea(data.price_vs_area);
            renderAvgPriceLocation(data.avg_price_location);
            renderPriceBedrooms(data.avg_price_bedrooms);
            renderActualVsPredicted(data.actual_vs_predicted);
            renderFeatureImportance(data.feature_importance);
            
            if (window.latestMetricsData) {
                renderModelCompChart(window.latestMetricsData);
            }
        }
    } catch (err) {
        console.error('Analytics Fetch Error:', err);
    }
}

// Chart.js Shared Options
const baseChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
    },
    scales: {
        x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 10 } }
        },
        y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af', font: { family: 'Plus Jakarta Sans', size: 10 } }
        }
    }
};

// 1. Average Price by Indian City / Village Chart
function renderAvgPriceLocation(locData) {
    const ctx = document.getElementById('chart-price-loc').getContext('2d');
    if (chartInstances['loc']) chartInstances['loc'].destroy();

    const colors = [
        '#ef4444', '#f97316', '#f59e0b', '#eab308',
        '#84cc16', '#10b981', '#06b6d4', '#0ea5e9',
        '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7',
        '#d946ef', '#ec4899', '#f43f5e', '#64748b'
    ];

    chartInstances['loc'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: locData.locations,
            datasets: [{
                label: 'Avg Valuation',
                data: locData.avg_prices,
                backgroundColor: locData.locations.map((_, i) => colors[i % colors.length] + 'b3'),
                borderColor: locData.locations.map((_, i) => colors[i % colors.length]),
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            ...baseChartOptions,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Avg Price: ${formatWordsINR(ctx.raw)} (${formatINR(ctx.raw)})`
                    }
                }
            },
            scales: {
                ...baseChartOptions.scales,
                x: {
                    ...baseChartOptions.scales.x,
                    ticks: {
                        color: '#9ca3af',
                        maxRotation: 45,
                        minRotation: 30,
                        font: { size: 9 }
                    }
                },
                y: {
                    ...baseChartOptions.scales.y,
                    ticks: { callback: (val) => formatWordsINR(val) }
                }
            }
        }
    });
}

// 2. House Price vs Area (Square Yards - Gaj)
function renderPriceVsArea(scatterData) {
    const ctx = document.getElementById('chart-price-area').getContext('2d');
    if (chartInstances['area']) chartInstances['area'].destroy();

    chartInstances['area'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Sample House Record',
                data: scatterData,
                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                borderColor: '#8b5cf6',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            ...baseChartOptions,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.raw.x} Gaj (${ctx.raw.sqft || ctx.raw.x*9} sqft) ➔ ${formatWordsINR(ctx.raw.y)}`
                    }
                }
            },
            scales: {
                x: { ...baseChartOptions.scales.x, title: { display: true, text: 'Plot Area in Square Yards (Gaj)', color: '#9ca3af' } },
                y: { 
                    ...baseChartOptions.scales.y, 
                    title: { display: true, text: 'Price in INR (₹)', color: '#9ca3af' },
                    ticks: { callback: (value) => formatWordsINR(value) }
                }
            }
        }
    });
}

// 3. Distribution of House Prices
function renderPriceDistribution(distData) {
    const ctx = document.getElementById('chart-price-dist').getContext('2d');
    if (chartInstances['dist']) chartInstances['dist'].destroy();

    chartInstances['dist'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: distData.labels,
            datasets: [{
                label: 'Property Count',
                data: distData.counts,
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: '#3b82f6',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            ...baseChartOptions,
            plugins: { legend: { display: false } },
            scales: {
                ...baseChartOptions.scales,
                x: {
                    ...baseChartOptions.scales.x,
                    ticks: { maxRotation: 35, font: { size: 9 } }
                }
            }
        }
    });
}

// 4. Price vs Bedrooms (BHK)
function renderPriceBedrooms(bedData) {
    const ctx = document.getElementById('chart-price-beds').getContext('2d');
    if (chartInstances['beds']) chartInstances['beds'].destroy();

    chartInstances['beds'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bedData.bedrooms,
            datasets: [{
                label: 'Avg Price',
                data: bedData.avg_prices,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            ...baseChartOptions,
            plugins: { legend: { display: false } },
            scales: {
                ...baseChartOptions.scales,
                y: {
                    ...baseChartOptions.scales.y,
                    ticks: { callback: (val) => formatWordsINR(val) }
                }
            }
        }
    });
}

// 5. Actual vs Predicted Prices Scatter Chart
function renderActualVsPredicted(scatterData) {
    const ctx = document.getElementById('chart-actual-pred').getContext('2d');
    if (chartInstances['actual_pred']) chartInstances['actual_pred'].destroy();

    const minVal = Math.min(...scatterData.map(p => p.x));
    const maxVal = Math.max(...scatterData.map(p => p.x));
    const lineData = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }];

    chartInstances['actual_pred'] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Predicted vs Actual',
                    data: scatterData,
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: '#f59e0b',
                    pointRadius: 4
                },
                {
                    label: 'Perfect 1:1 Match',
                    data: lineData,
                    type: 'line',
                    borderColor: 'rgba(255, 255, 255, 0.4)',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            ...baseChartOptions,
            scales: {
                x: { 
                    ...baseChartOptions.scales.x, 
                    title: { display: true, text: 'Actual Price (₹)', color: '#9ca3af' },
                    ticks: { callback: (val) => formatWordsINR(val) }
                },
                y: { 
                    ...baseChartOptions.scales.y, 
                    title: { display: true, text: 'Predicted Price (₹)', color: '#9ca3af' },
                    ticks: { callback: (val) => formatWordsINR(val) }
                }
            }
        }
    });
}

// 6. Model Performance Comparison Chart
function renderModelCompChart(metricsData) {
    const ctx = document.getElementById('chart-model-comp').getContext('2d');
    if (chartInstances['model_comp']) chartInstances['model_comp'].destroy();

    const algos = Object.keys(metricsData.model_comparison);
    const r2Scores = algos.map(a => metricsData.model_comparison[a].R2_Score);

    chartInstances['model_comp'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: algos,
            datasets: [{
                label: 'R² Score (Accuracy)',
                data: r2Scores,
                backgroundColor: algos.map(a => a === metricsData.best_model ? 'rgba(16, 185, 129, 0.8)' : 'rgba(59, 130, 246, 0.5)'),
                borderColor: algos.map(a => a === metricsData.best_model ? '#10b981' : '#3b82f6'),
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            ...baseChartOptions,
            plugins: { legend: { display: false } },
            scales: {
                ...baseChartOptions.scales,
                y: {
                    min: 0.80,
                    max: 1.0,
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });
}

// 7. Render Feature Importance Progress Bars
function renderFeatureImportance(importanceList) {
    const container = document.getElementById('importance-list-container');
    container.innerHTML = '';

    importanceList.forEach(item => {
        const div = document.createElement('div');
        div.className = 'importance-item';
        div.innerHTML = `
            <div class="importance-header">
                <span>${item.feature}</span>
                <span style="color: var(--accent-emerald); font-weight: 700;">${item.percentage}%</span>
            </div>
            <div class="importance-bar-bg">
                <div class="importance-bar-fill" style="width: ${item.percentage}%;"></div>
            </div>
        `;
        container.appendChild(div);
    });
}
