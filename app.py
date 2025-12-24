"""
En iyi modelin kullanıcı arayüzü - predict ve predict_proba gösterimi
URL Analizi ile Phishing Tespiti
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import re
from urllib.parse import urlparse
import socket

app = Flask(__name__)
CORS(app)


def extract_url_features(url):
    """URL'den özellik çıkarma fonksiyonu"""
    features = {}
    
    # URL'yi parse et
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'http://' + url
            parsed = urlparse(url)
    except:
        parsed = urlparse('http://example.com')
    
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    # 1. IP Adresi kontrolü (-1: IP var, 1: IP yok)
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    has_ip = -1 if re.search(ip_pattern, domain) else 1
    features['having_IPhaving_IP_Address'] = has_ip
    
    # 2. URL Uzunluğu (-1: >75, 0: 54-75, 1: <54)
    url_len = len(url)
    if url_len < 54:
        features['URLURL_Length'] = 1
    elif url_len <= 75:
        features['URLURL_Length'] = 0
    else:
        features['URLURL_Length'] = -1
    
    # 3. Kısaltma servisi kontrolü (-1: var, 1: yok)
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']
    has_shortening = -1 if any(s in url.lower() for s in shortening_services) else 1
    features['Shortining_Service'] = has_shortening
    
    # 4. @ sembolü (-1: var, 1: yok)
    features['having_At_Symbol'] = -1 if '@' in url else 1
    
    # 5. Çift slash yönlendirme (-1: şüpheli, 1: normal)
    double_slash_pos = url.find('//')
    if double_slash_pos > 7:  # http:// sonrasında // varsa
        features['double_slash_redirecting'] = -1
    else:
        features['double_slash_redirecting'] = 1
    
    # 6. Prefix-Suffix (-1: tire var, 1: tire yok)
    features['Prefix_Suffix'] = -1 if '-' in domain else 1
    
    # 7. Subdomain sayısı (-1: >3, 0: 2-3, 1: 1)
    subdomain_count = domain.count('.')
    if subdomain_count == 1:
        features['having_Sub_Domain'] = 1
    elif subdomain_count == 2:
        features['having_Sub_Domain'] = 0
    else:
        features['having_Sub_Domain'] = -1
    
    # 8. SSL (HTTPS) (-1: yok/güvensiz, 0: şüpheli, 1: güvenli)
    if parsed.scheme == 'https':
        features['SSLfinal_State'] = 1
    else:
        features['SSLfinal_State'] = -1
    
    # 9. Domain kayıt süresi (varsayılan: belirsiz)
    features['Domain_registeration_length'] = 0
    
    # 10. Favicon (varsayılan: normal)
    features['Favicon'] = 1
    
    # 11. Port (varsayılan: normal)
    features['port'] = 1
    
    # 12. HTTPS token (-1: domain'de https var, 1: yok)
    features['HTTPS_token'] = -1 if 'https' in domain.lower() else 1
    
    # 13. Request URL (varsayılan)
    features['Request_URL'] = 1
    
    # 14. URL of Anchor (varsayılan)
    features['URL_of_Anchor'] = 0
    
    # 15. Links in tags (varsayılan)
    features['Links_in_tags'] = 0
    
    # 16. SFH (varsayılan)
    features['SFH'] = 1
    
    # 17. Email submit (varsayılan: yok)
    features['Submitting_to_email'] = 1
    
    # 18. Abnormal URL (-1: anormal, 1: normal)
    features['Abnormal_URL'] = 1
    
    # 19. Redirect (varsayılan)
    features['Redirect'] = 0
    
    # 20. on_mouseover (varsayılan: yok)
    features['on_mouseover'] = 1
    
    # 21. RightClick (varsayılan: aktif)
    features['RightClick'] = 1
    
    # 22. popUpWindow (varsayılan: yok)
    features['popUpWidnow'] = 1
    
    # 23. Iframe (varsayılan: yok)
    features['Iframe'] = 1
    
    # 24. Domain yaşı (varsayılan: belirsiz)
    features['age_of_domain'] = 0
    
    # 25. DNS Record (varsayılan: var)
    features['DNSRecord'] = 1
    
    # 26. Web traffic (varsayılan: orta)
    features['web_traffic'] = 0
    
    # 27. Page Rank (varsayılan: orta)
    features['Page_Rank'] = 0
    
    # 28. Google Index (varsayılan: var)
    features['Google_Index'] = 1
    
    # 29. Links pointing to page (varsayılan: orta)
    features['Links_pointing_to_page'] = 0
    
    # 30. Statistical report (varsayılan: yok)
    features['Statistical_report'] = 1
    
    # Şüpheli kelime kontrolü (ek bilgi için)
    suspicious_words = ['login', 'signin', 'verify', 'account', 'update', 'secure', 
                        'banking', 'confirm', 'password', 'credential', 'suspend']
    has_suspicious = any(word in url.lower() for word in suspicious_words)
    features['has_suspicious_words'] = has_suspicious
    
    # Ek URL istatistikleri (analiz için)
    features['url_stats'] = {
        'url_length': len(url),
        'domain': domain,
        'path': path,
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'has_https': parsed.scheme == 'https',
        'has_ip': bool(re.search(ip_pattern, domain)),
        'subdomain_count': subdomain_count
    }
    
    return features

# Model ve scaler yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

# Global değişkenler
model = None
scaler = None
model_name = "Yükleniyor..."

# Özellik isimleri (Kaggle Dataset - UCI Phishing Websites)
FEATURE_NAMES = [
    'having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service',
    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
    'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
    'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]


def load_model():
    """En iyi modeli yükle"""
    global model, scaler, model_name
    
    try:
        # Önce en iyi modeli dene
        best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        
        if os.path.exists(best_model_path):
            model = joblib.load(best_model_path)
            model_name = "En İyi Model"
            print(f"Model yüklendi: {best_model_path}")
        else:
            # Alternatif olarak Random Forest'ı dene
            rf_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
            if os.path.exists(rf_path):
                model = joblib.load(rf_path)
                model_name = "Random Forest"
                print(f"Model yüklendi: {rf_path}")
            else:
                print("UYARI: Model dosyası bulunamadı!")
                model = None
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Scaler yüklendi: {scaler_path}")
        else:
            print("UYARI: Scaler dosyası bulunamadı!")
            scaler = None
            
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        model = None
        scaler = None


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html', 
                         feature_names=FEATURE_NAMES,
                         model_name=model_name)


@app.route('/predict', methods=['POST'])
def predict():
    """Tahmin endpoint'i"""
    global model, scaler
    
    try:
        # Form verisini al
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Veri bulunamadı',
                'success': False
            }), 400
        
        # Özellikleri al
        features = []
        for feature_name in FEATURE_NAMES:
            value = data.get(feature_name, 0)
            try:
                features.append(float(value))
            except:
                features.append(0)
        
        features = np.array(features).reshape(1, -1)
        
        # Model yüklü mü kontrol et
        if model is None:
            # Demo modu - basit bir kural tabanlı tahmin
            url_length = features[0, 0]
            has_https = features[0, 1]
            has_ip = features[0, 2]
            
            # Basit skor hesapla
            risk_score = 0
            risk_score += 0.3 if url_length > 75 else 0
            risk_score += 0.25 if has_https == 0 else 0
            risk_score += 0.35 if has_ip == 1 else 0
            risk_score += np.random.uniform(0, 0.1)  # Küçük rastgelelik
            
            phishing_prob = min(max(risk_score, 0.05), 0.95)
            prediction = 1 if phishing_prob > 0.5 else 0
            
            return jsonify({
                'success': True,
                'prediction': int(prediction),
                'prediction_label': 'Phishing (Zararlı)' if prediction == 1 else 'Legitimate (Güvenli)',
                'probability_phishing': round(phishing_prob * 100, 2),
                'probability_legitimate': round((1 - phishing_prob) * 100, 2),
                'model_name': 'Demo Modu (Model yüklenmedi)',
                'warning': 'Gerçek model yüklenmedi, demo sonuçlar gösteriliyor.'
            })
        
        # Ölçeklendirme
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Tahmin
        prediction = model.predict(features_scaled)[0]
        
        # Olasılık tahmini
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            prob_legitimate = probabilities[0]
            prob_phishing = probabilities[1]
        else:
            prob_phishing = 1.0 if prediction == 1 else 0.0
            prob_legitimate = 1.0 - prob_phishing
        
        # Sonuç
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Phishing (Zararlı)' if prediction == 1 else 'Legitimate (Güvenli)',
            'probability_phishing': round(prob_phishing * 100, 2),
            'probability_legitimate': round(prob_legitimate * 100, 2),
            'model_name': model_name,
            'features_used': len(FEATURE_NAMES)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/features')
def get_features():
    """Özellik listesini döndür"""
    return jsonify({
        'features': FEATURE_NAMES,
        'count': len(FEATURE_NAMES)
    })


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """URL analiz endpoint'i - URL'den özellikleri çıkar ve tahmin yap"""
    global model, scaler
    
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({
                'error': 'URL adresi girilmedi',
                'success': False
            }), 400
        
        # URL'den özellikleri çıkar
        extracted = extract_url_features(url)
        url_stats = extracted.pop('url_stats', {})
        has_suspicious = extracted.pop('has_suspicious_words', False)
        
        # Özellik vektörünü oluştur
        features = []
        for feature_name in FEATURE_NAMES:
            value = extracted.get(feature_name, 0)
            features.append(float(value))
        
        features = np.array(features).reshape(1, -1)
        
        # Model yüklü mü kontrol et
        if model is None:
            # Demo modu - URL özelliklerine göre basit tahmin
            risk_score = 0
            
            # IP adresi varsa
            if extracted.get('having_IPhaving_IP_Address') == -1:
                risk_score += 0.25
            
            # HTTPS yoksa
            if extracted.get('SSLfinal_State') == -1:
                risk_score += 0.2
            
            # URL çok uzunsa
            if extracted.get('URLURL_Length') == -1:
                risk_score += 0.15
            
            # Subdomain fazlaysa
            if extracted.get('having_Sub_Domain') == -1:
                risk_score += 0.15
            
            # @ sembolü varsa
            if extracted.get('having_At_Symbol') == -1:
                risk_score += 0.1
            
            # Şüpheli kelime varsa
            if has_suspicious:
                risk_score += 0.15
            
            phishing_prob = min(max(risk_score, 0.05), 0.95)
            prediction = 1 if phishing_prob > 0.5 else 0
            
            return jsonify({
                'success': True,
                'url': url,
                'prediction': int(prediction),
                'prediction_label': 'Phishing (Zararlı)' if prediction == 1 else 'Legitimate (Güvenli)',
                'probability_phishing': round(phishing_prob * 100, 2),
                'probability_legitimate': round((1 - phishing_prob) * 100, 2),
                'model_name': 'Demo Modu (Model yüklenmedi)',
                'features_extracted': extracted,
                'url_stats': url_stats,
                'has_suspicious_words': has_suspicious,
                'warning': 'Gerçek model yüklenmedi, demo sonuçlar gösteriliyor.'
            })
        
        # Ölçeklendirme
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Tahmin
        prediction = model.predict(features_scaled)[0]
        
        # Olasılık tahmini
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            prob_legitimate = probabilities[0]
            prob_phishing = probabilities[1]
        else:
            prob_phishing = 1.0 if prediction == 1 else 0.0
            prob_legitimate = 1.0 - prob_phishing
        
        # Sonuç
        return jsonify({
            'success': True,
            'url': url,
            'prediction': int(prediction),
            'prediction_label': 'Phishing (Zararlı)' if prediction == 1 else 'Legitimate (Güvenli)',
            'probability_phishing': round(prob_phishing * 100, 2),
            'probability_legitimate': round(prob_legitimate * 100, 2),
            'model_name': model_name,
            'features_extracted': extracted,
            'url_stats': url_stats,
            'has_suspicious_words': has_suspicious,
            'features_used': len(FEATURE_NAMES)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/model-info')
def get_model_info():
    """Model bilgilerini döndür"""
    return jsonify({
        'model_name': model_name,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_count': len(FEATURE_NAMES)
    })


if __name__ == '__main__':
    # Modeli yükle
    load_model()
    
    print("\n" + "="*50)
    print("PHİSHİNG TESPİT WEB UYGULAMASI")
    print("224410054_SenanurOzbag_Proje")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Model yüklendi: {'Evet' if model else 'Hayır (Demo modu)'}")
    print(f"Scaler yüklendi: {'Evet' if scaler else 'Hayır'}")
    print("="*50)
    print("\nUygulama başlatılıyor: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
