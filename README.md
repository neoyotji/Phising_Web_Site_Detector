# 🛡️ Phishing Web Sitesi Tespit Sistemi

> **Makine Öğrenmesi ile Kimlik Avı (Phishing) Web Sitelerinin Tespiti**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-yellow.svg)](https://scikit-learn.org)

![Phishing Detection](https://img.shields.io/badge/Security-Phishing%20Detection-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Proje Hakkında

Bu proje, makine öğrenmesi ve derin öğrenme algoritmaları kullanarak potansiyel **phishing (kimlik avı) web sitelerini** tespit eden bir sistemdir. Kullanıcı dostu web arayüzü sayesinde, herhangi bir URL adresinin güvenliğini kolayca analiz edebilirsiniz.

### 🎯 Proje Amacı

Günümüzde siber güvenlik tehditlerinin başında gelen phishing saldırıları, kullanıcılar için ciddi riskler oluşturmaktadır. Bu projenin temel amacı:

- URL özelliklerinden anlamlı özellikler çıkarmak
- Farklı makine öğrenmesi modellerini karşılaştırmak
- En yüksek doğrulukla phishing sitelerini tespit etmek
- Kullanıcı dostu bir web arayüzü sunmak

## ✨ Özellikler

- 🔍 **Hızlı URL Analizi**: Sadece URL girerek anında analiz
- 📊 **30 Farklı Özellik**: Kapsamlı URL özellik çıkarımı
- 🤖 **Çoklu Model Desteği**: Random Forest, SVM, Neural Network ve daha fazlası
- 📈 **Olasılık Gösterimi**: Phishing/Legitimate olasılık yüzdeleri
- 🎨 **Modern Web Arayüzü**: Responsive ve kullanıcı dostu tasarım
- ⚡ **Gerçek Zamanlı Tahmin**: Anlık sonuç alma

## 🏗️ Proje Yapısı

```
📦 Phising_Web_Site_Detector
├── 📂 app.py                    # Flask web uygulaması
├── 📂 models/                   # ML model eğitim scriptleri
│   ├── classifiers.py          # Sınıflandırma modelleri
│   ├── neural_network.py       # Yapay sinir ağı modeli
│   ├── data_preprocessing.py   # Veri ön işleme
│   ├── statistical_tests.py    # İstatistiksel testler
│   ├── evaluation.py           # Model değerlendirme
│   └── main.py                 # Ana eğitim scripti
├── 📂 saved_models/             # Eğitilmiş modeller (.pkl)
├── 📂 data/                     # Veri setleri
│   ├── dataset.csv
│   └── phishing_dataset.csv
├── 📂 templates/                # HTML şablonları
│   └── index.html
├── 📂 static/                   # CSS ve JavaScript
│   ├── style.css
│   └── script.js
├── 📂 outputs/                  # Çıktılar ve grafikler
├── 📄 requirements.txt          # Bağımlılıklar
└── 📄 README.md
```

## 🔧 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### Adım Adım Kurulum

1. **Projeyi klonlayın**
```bash
git clone https://github.com/neoyotji/Phising_Web_Site_Detector.git
cd Phising_Web_Site_Detector
```

2. **Sanal ortam oluşturun (önerilen)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Bağımlılıkları yükleyin**
```bash
pip install -r requirements.txt
```

4. **Uygulamayı başlatın**
```bash
python app.py
```

5. **Tarayıcıda açın**
```
http://localhost:5000
```

## 🚀 Kullanım

### Web Arayüzü ile

1. Uygulamayı başlattıktan sonra tarayıcınızda `http://localhost:5000` adresine gidin
2. **Hızlı URL Analizi** bölümüne analiz etmek istediğiniz URL'yi girin
3. **"Analiz Et"** butonuna tıklayın
4. Sonuçları görüntüleyin:
   - ✅ **Güvenli (Legitimate)**: Site güvenli görünüyor
   - ❌ **Phishing (Zararlı)**: Site potansiyel tehlike içeriyor

### Örnek URL'ler

| URL Tipi | Örnek |
|----------|-------|
| Güvenli | `https://www.google.com` |
| Şüpheli IP | `http://192.168.1.1/login/verify` |
| Phishing | `https://secure-login-bank.suspicious-site.com/account/verify` |

## 🤖 Kullanılan Modeller

| Model | Açıklama |
|-------|----------|
| **Random Forest** | Ensemble öğrenme yöntemi |
| **SVM (Support Vector Machine)** | Destek vektör makineleri |
| **Decision Tree** | Karar ağaçları |
| **K-Nearest Neighbors (KNN)** | En yakın komşu algoritması |
| **Naive Bayes** | Olasılıksal sınıflandırıcı |
| **Neural Network** | TensorFlow ile yapay sinir ağı |

## 📊 Analiz Edilen Özellikler

Sistem, URL'lerden **30 farklı özellik** çıkararak analiz yapar:

### URL Özellikleri
- IP Adresi varlığı
- URL uzunluğu
- Kısaltma servisi kullanımı (bit.ly, tinyurl vb.)
- @ sembolü varlığı
- Çift slash yönlendirme
- Prefix/Suffix (tire işareti)
- Subdomain sayısı

### Güvenlik Özellikleri
- SSL/HTTPS durumu
- HTTPS token domain'de
- Favicon kaynağı
- Port kullanımı

### Domain Özellikleri
- Domain kayıt süresi
- Domain yaşı
- DNS kaydı varlığı
- WHOIS bilgileri

### Sayfa Özellikleri
- Harici URL oranı
- Anchor URL'leri
- Links in tags
- SFH (Server Form Handler)
- E-mail gönderimi
- Redirect sayısı
- Pop-up pencereler
- Iframe kullanımı

### İstatistiksel Özellikler
- Web trafiği
- Page Rank
- Google Index durumu
- Sayfaya gelen link sayısı
- İstatistiksel raporlarda varlık

## 🛠️ Teknoloji Yığını

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing

### Machine Learning
- **scikit-learn** - ML algoritmaları
- **TensorFlow** - Derin öğrenme
- **NumPy & Pandas** - Veri işleme

### İstatistik
- **SciPy** - Bilimsel hesaplamalar
- **statsmodels** - İstatistiksel modeller

### Görselleştirme
- **Matplotlib** - Grafik oluşturma
- **Seaborn** - İstatistiksel görselleştirme

### Frontend
- **HTML5 & CSS3** - Modern web tasarım
- **JavaScript** - Dinamik arayüz
- **Google Fonts (Inter)** - Tipografi

## 📈 Model Performansı

Modeller, UCI Machine Learning Repository'den alınan Phishing Websites veri seti ile eğitilmiştir.

## 📁 Veri Seti

Proje, Kaggle ve UCI ML Repository'den alınan phishing web sitesi veri setlerini kullanmaktadır:
- `dataset.csv` - Ana veri seti
- `phishing_dataset.csv` - Kaggle phishing veri seti

## 🔒 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana sayfa |
| `/analyze-url` | POST | URL analizi |
| `/predict` | POST | Manuel özellik tahmini |
| `/api/features` | GET | Özellik listesi |
| `/api/model-info` | GET | Model bilgileri |

## 👩‍💻 Geliştirici

**Senanur Özbağ**
- 📧 Öğrenci No: 224410054
- 📚 Ders: Makine Öğrenmesi

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 🙏 Teşekkürler

- UCI Machine Learning Repository - Veri seti için
- Kaggle - Phishing veri seti için
- Flask & scikit-learn toplulukları

---

<p align="center">
  <b>🛡️ Phishing Tespit Sistemi - 2024</b><br>
  <i>Makine Öğrenmesi ile Siber Güvenlik</i>
</p>
