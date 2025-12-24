/**
 * Phishing Web Sitesi Tespiti - JavaScript
 * 224410054_SenanurOzbag_Proje
 * URL Analizi ve Manuel Form Desteği
 */

// DOM Elements - URL Analiz
const urlInput = document.getElementById('urlInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const urlResultsSection = document.getElementById('urlResultsSection');
const urlLoading = document.getElementById('urlLoading');
const urlResultCard = document.getElementById('urlResultCard');

// DOM Elements - Manuel Form
const predictionForm = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const resultsSection = document.getElementById('resultsSection');
const toggleManualForm = document.getElementById('toggleManualForm');
const manualFormSection = document.getElementById('manualFormSection');

// Feature names (same order as backend)
const FEATURE_NAMES = [
    'url_length', 'has_https', 'has_ip_address', 'num_dots', 'num_hyphens',
    'num_underscores', 'num_slashes', 'num_questionmarks', 'num_equals',
    'num_at_symbols', 'num_ampersands', 'num_digits', 'domain_length',
    'path_length', 'subdomain_count', 'has_suspicious_words', 'domain_age_days',
    'alexa_rank', 'google_index', 'page_rank', 'dns_record', 'web_traffic',
    'links_in_tags', 'links_pointing_to_page', 'statistical_report'
];

/**
 * URL Analiz - Butona tıklama
 */
analyzeBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();

    if (!url) {
        alert('Lütfen bir URL adresi girin!');
        urlInput.focus();
        return;
    }

    await analyzeUrl(url);
});

/**
 * URL Input - Enter tuşu
 */
urlInput.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        analyzeBtn.click();
    }
});

/**
 * Örnek URL butonları
 */
document.querySelectorAll('.example-url-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        urlInput.value = btn.dataset.url;
        urlInput.focus();
    });
});

/**
 * URL Analiz fonksiyonu
 */
async function analyzeUrl(url) {
    // Show loading
    urlResultsSection.style.display = 'block';
    urlLoading.classList.add('active');
    urlResultCard.classList.remove('active');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-small"></span> Analiz ediliyor...';

    try {
        const response = await fetch('/analyze-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url })
        });

        const result = await response.json();

        if (result.success) {
            displayUrlResult(result);
        } else {
            showUrlError(result.error || 'Bir hata oluştu');
        }
    } catch (error) {
        console.error('API Error:', error);
        showUrlError('Sunucu ile bağlantı kurulamadı. Lütfen tekrar deneyin.');
    }

    // Reset button
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-text">Analiz Et</span>';
}

/**
 * URL Analiz sonucunu göster
 */
function displayUrlResult(result) {
    urlLoading.classList.remove('active');

    // Analiz edilen URL
    document.getElementById('analyzedUrl').innerHTML = `
        <span class="analyzed-label">Analiz Edilen URL:</span>
        <span class="analyzed-value">${escapeHtml(result.url)}</span>
    `;

    const resultStatus = document.getElementById('urlResultStatus');
    const statusIcon = document.getElementById('urlStatusIcon');
    const predictionLabel = document.getElementById('urlPredictionLabel');
    const confidenceText = document.getElementById('urlConfidenceText');
    const probSafe = document.getElementById('urlProbSafe');
    const probDanger = document.getElementById('urlProbDanger');
    const probBarSafe = document.getElementById('urlProbBarSafe');
    const probBarDanger = document.getElementById('urlProbBarDanger');
    const usedModel = document.getElementById('urlUsedModel');
    const predictValue = document.getElementById('urlPredictValue');
    const featureCount = document.getElementById('urlFeatureCount');
    const warningBox = document.getElementById('urlWarningBox');
    const warningText = document.getElementById('urlWarningText');

    // Determine if safe or dangerous
    const isSafe = result.prediction === 0;
    const maxProb = Math.max(result.probability_legitimate, result.probability_phishing);

    // Update status
    resultStatus.className = 'result-status ' + (isSafe ? 'safe' : 'danger');
    statusIcon.textContent = isSafe ? '✓' : '⚠';
    predictionLabel.textContent = result.prediction_label;
    confidenceText.textContent = `%${maxProb.toFixed(1)} olasılıkla ${isSafe ? 'güvenli' : 'zararlı'}`;

    // Update probability bars
    probSafe.textContent = `%${result.probability_legitimate.toFixed(1)}`;
    probDanger.textContent = `%${result.probability_phishing.toFixed(1)}`;

    // Animate bars
    setTimeout(() => {
        probBarSafe.style.width = `${result.probability_legitimate}%`;
        probBarDanger.style.width = `${result.probability_phishing}%`;
    }, 100);

    // Update details
    usedModel.textContent = result.model_name || '-';
    predictValue.textContent = result.prediction === 1 ? '1 (Phishing)' : '0 (Legitimate)';
    featureCount.textContent = result.features_used || '30';

    // URL özellikleri tablosu
    displayUrlFeatures(result.url_stats, result.features_extracted, result.has_suspicious_words);

    // Show warning if demo mode
    if (result.warning) {
        warningBox.style.display = 'flex';
        warningText.textContent = result.warning;
    } else {
        warningBox.style.display = 'none';
    }

    // Show result card
    urlResultCard.classList.add('active');

    // Scroll to results
    urlResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * URL özelliklerini göster
 */
function displayUrlFeatures(stats, features, hasSuspicious) {
    const grid = document.getElementById('urlFeaturesGrid');

    const featureItems = [
        { label: 'URL Uzunlugu', value: stats.url_length, icon: '', status: stats.url_length > 75 ? 'danger' : 'safe' },
        { label: 'Domain', value: stats.domain, icon: '', status: 'neutral' },
        { label: 'HTTPS', value: stats.has_https ? 'Evet' : 'Hayir', icon: '', status: stats.has_https ? 'safe' : 'danger' },
        { label: 'IP Adresi', value: stats.has_ip ? 'Var' : 'Yok', icon: '', status: stats.has_ip ? 'danger' : 'safe' },
        { label: 'Subdomain Sayisi', value: stats.subdomain_count, icon: '', status: stats.subdomain_count > 2 ? 'warning' : 'safe' },
        { label: 'Nokta Sayisi', value: stats.num_dots, icon: '', status: stats.num_dots > 4 ? 'warning' : 'safe' },
        { label: 'Tire Sayisi', value: stats.num_hyphens, icon: '', status: stats.num_hyphens > 3 ? 'warning' : 'safe' },
        { label: 'Slash Sayisi', value: stats.num_slashes, icon: '', status: 'neutral' },
        { label: 'Rakam Sayisi', value: stats.num_digits, icon: '', status: stats.num_digits > 10 ? 'warning' : 'safe' },
        { label: 'Supheli Kelime', value: hasSuspicious ? 'Var' : 'Yok', icon: '', status: hasSuspicious ? 'danger' : 'safe' }
    ];

    grid.innerHTML = featureItems.map(item => `
        <div class="url-feature-item ${item.status}">
            <span class="feature-label">${item.label}</span>
            <span class="feature-value">${item.value}</span>
        </div>
    `).join('');
}

/**
 * URL Analiz hatası göster
 */
function showUrlError(message) {
    urlLoading.classList.remove('active');

    const resultStatus = document.getElementById('urlResultStatus');
    const statusIcon = document.getElementById('urlStatusIcon');
    const predictionLabel = document.getElementById('urlPredictionLabel');
    const confidenceText = document.getElementById('urlConfidenceText');

    resultStatus.className = 'result-status danger';
    statusIcon.textContent = '❌';
    predictionLabel.textContent = 'Hata Oluştu';
    confidenceText.textContent = message;

    // Hide probability bars
    document.getElementById('urlProbBarSafe').style.width = '0%';
    document.getElementById('urlProbBarDanger').style.width = '0%';
    document.getElementById('urlProbSafe').textContent = '-%';
    document.getElementById('urlProbDanger').textContent = '-%';

    urlResultCard.classList.add('active');
}

/**
 * HTML escape
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Toggle manuel form görünürlüğü
 */
toggleManualForm.addEventListener('click', () => {
    const form = predictionForm;
    const icon = toggleManualForm.querySelector('.toggle-icon');

    if (form.style.display === 'none') {
        form.style.display = 'block';
        manualFormSection.classList.remove('collapsed');
        icon.textContent = '▲';
    } else {
        form.style.display = 'none';
        manualFormSection.classList.add('collapsed');
        icon.textContent = '▼';
    }
});

/**
 * Manuel Form submit handler
 */
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = {};
    FEATURE_NAMES.forEach(name => {
        const element = document.getElementById(name);
        if (element) {
            formData[name] = parseFloat(element.value) || 0;
        } else {
            formData[name] = 0;
        }
    });

    // Show loading
    showLoading();

    try {
        // API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
        } else {
            showError(result.error || 'Bir hata oluştu');
        }
    } catch (error) {
        console.error('API Error:', error);
        showError('Sunucu ile bağlantı kurulamadı. Lütfen tekrar deneyin.');
    }
});

/**
 * Show loading state
 */
function showLoading() {
    resultsSection.style.display = 'block';
    loading.classList.add('active');
    resultCard.classList.remove('active');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-small"></span> Analiz ediliyor...';
}

/**
 * Hide loading state
 */
function hideLoading() {
    loading.classList.remove('active');
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-text">Manuel Analiz Et</span>';
}

/**
 * Display prediction result
 */
function displayResult(result) {
    hideLoading();

    const resultStatus = document.getElementById('resultStatus');
    const statusIcon = document.getElementById('statusIcon');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceText = document.getElementById('confidenceText');
    const probSafe = document.getElementById('probSafe');
    const probDanger = document.getElementById('probDanger');
    const probBarSafe = document.getElementById('probBarSafe');
    const probBarDanger = document.getElementById('probBarDanger');
    const usedModel = document.getElementById('usedModel');
    const predictValue = document.getElementById('predictValue');
    const featureCount = document.getElementById('featureCount');
    const warningBox = document.getElementById('warningBox');
    const warningText = document.getElementById('warningText');

    // Determine if safe or dangerous
    const isSafe = result.prediction === 0;
    const maxProb = Math.max(result.probability_legitimate, result.probability_phishing);

    // Update status
    resultStatus.className = 'result-status ' + (isSafe ? 'safe' : 'danger');
    statusIcon.textContent = isSafe ? '✓' : '⚠';
    predictionLabel.textContent = result.prediction_label;
    confidenceText.textContent = `%${maxProb.toFixed(1)} olasılıkla ${isSafe ? 'güvenli' : 'zararlı'}`;

    // Update probability bars
    probSafe.textContent = `%${result.probability_legitimate.toFixed(1)}`;
    probDanger.textContent = `%${result.probability_phishing.toFixed(1)}`;

    // Animate bars
    setTimeout(() => {
        probBarSafe.style.width = `${result.probability_legitimate}%`;
        probBarDanger.style.width = `${result.probability_phishing}%`;
    }, 100);

    // Update details
    usedModel.textContent = result.model_name || '-';
    predictValue.textContent = result.prediction === 1 ? '1 (Phishing)' : '0 (Legitimate)';
    featureCount.textContent = result.features_used || FEATURE_NAMES.length;

    // Show warning if demo mode
    if (result.warning) {
        warningBox.style.display = 'flex';
        warningText.textContent = result.warning;
    } else {
        warningBox.style.display = 'none';
    }

    // Show result card
    resultCard.classList.add('active');

    // Scroll to results on mobile
    if (window.innerWidth <= 1024) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * Show error message
 */
function showError(message) {
    hideLoading();

    const resultStatus = document.getElementById('resultStatus');
    const statusIcon = document.getElementById('statusIcon');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceText = document.getElementById('confidenceText');

    resultStatus.className = 'result-status danger';
    statusIcon.textContent = '❌';
    predictionLabel.textContent = 'Hata Oluştu';
    confidenceText.textContent = message;

    // Hide probability bars
    document.getElementById('probBarSafe').style.width = '0%';
    document.getElementById('probBarDanger').style.width = '0%';
    document.getElementById('probSafe').textContent = '-%';
    document.getElementById('probDanger').textContent = '-%';

    resultCard.classList.add('active');
}

/**
 * Set example values for quick testing
 */
function setExampleValues(type) {
    if (type === 'safe') {
        // Typical legitimate website
        document.getElementById('url_length').value = 45;
        document.getElementById('has_https').value = 1;
        document.getElementById('has_ip_address').value = 0;
        document.getElementById('num_dots').value = 2;
        document.getElementById('num_hyphens').value = 1;
        document.getElementById('domain_age_days').value = 1500;
        document.getElementById('alexa_rank').value = 50000;
        document.getElementById('google_index').value = 1;
        document.getElementById('page_rank').value = 6;
        document.getElementById('has_suspicious_words').value = 0;
    } else if (type === 'phishing') {
        // Typical phishing website characteristics
        document.getElementById('url_length').value = 120;
        document.getElementById('has_https').value = 0;
        document.getElementById('has_ip_address').value = 1;
        document.getElementById('num_dots').value = 6;
        document.getElementById('num_hyphens').value = 4;
        document.getElementById('domain_age_days').value = 30;
        document.getElementById('alexa_rank').value = 800000;
        document.getElementById('google_index').value = 0;
        document.getElementById('page_rank').value = 1;
        document.getElementById('has_suspicious_words').value = 1;
    }
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('Phishing Detection App initialized');
    console.log('Features:', FEATURE_NAMES.length);

    // URL input'a odaklan
    urlInput.focus();

    // Add keyboard shortcut for submit
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            if (document.activeElement === urlInput) {
                analyzeBtn.click();
            } else {
                predictionForm.dispatchEvent(new Event('submit'));
            }
        }
    });
});
