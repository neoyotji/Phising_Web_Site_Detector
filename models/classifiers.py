"""
5 Sınıflandırıcı + 2 Topluluk Öğrenmesi Yaklaşımı
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import joblib
import os


class PhishingClassifiers:
    """5 temel sınıflandırıcı ve 2 topluluk öğrenmesi modeli"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Tüm modelleri başlat - Overfitting önleme ile"""
        
        # 1. Logistic Regression (L2 regularization ile)
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs',
            C=0.5  # Daha güçlü regularization
        )
        
        # 2. Decision Tree (Overfitting önleme: daha sığ ağaç, daha fazla pruning)
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=8,  # 10 -> 8 (daha sığ)
            min_samples_split=10,  # 5 -> 10 (daha az bölme)
            min_samples_leaf=5,  # Yaprakta minimum örnek
            random_state=self.random_state
        )
        
        # 3. Random Forest (Overfitting önleme: daha sığ ağaçlar)
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # 15 -> 10 (daha sığ)
            min_samples_split=10,  # 5 -> 10
            min_samples_leaf=4,  # Yaprakta minimum örnek
            max_features='sqrt',  # Özellik alt kümesi
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 4. Support Vector Machine 
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=0.8,  # 1.0 -> 0.8 (daha güçlü regularization)
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        # 5. K-Nearest Neighbors (KNN) - OVERFITTING FIX!
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=15,  # 5 -> 15 (daha fazla komşu = daha az overfitting)
            weights='uniform',  # 'distance' -> 'uniform' (daha basit)
            metric='euclidean',
            n_jobs=-1
        )
        
        # === TOPLULUK ÖĞRENMESİ ===
        
        # 6. Voting Classifier (Soft Voting) - Güncellenmiş parametrelerle
        self.models['Voting Classifier'] = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, C=0.5, random_state=self.random_state)),
                ('dt', DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=self.random_state)),
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.random_state)),
                ('svm', SVC(probability=True, C=0.8, random_state=self.random_state)),
                ('knn', KNeighborsClassifier(n_neighbors=15, weights='uniform'))
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # 7. Gradient Boosting (Overfitting önleme: daha düşük learning rate, daha sığ)
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,  # 0.1 -> 0.08 (daha yavaş öğrenme)
            max_depth=4,  # 5 -> 4 (daha sığ)
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,  # Alt örnekleme (regularization)
            random_state=self.random_state
        )
    
    def get_model_names(self):
        """Model isimlerini döndür"""
        return list(self.models.keys())
    
    def get_model(self, name):
        """Belirli bir modeli döndür"""
        return self.models.get(name)
    
    def train_model(self, name, X_train, y_train):
        """Belirli bir modeli eğit"""
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
        
        print(f"\n{'='*50}")
        print(f"Model Eğitiliyor: {name}")
        print(f"{'='*50}")
        
        model = self.models[name]
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        
        # Training accuracy
        train_score = model.score(X_train, y_train)
        print(f"Eğitim Doğruluğu: {train_score:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """Tüm modelleri eğit"""
        results = {}
        
        for name in self.models:
            try:
                model = self.train_model(name, X_train, y_train)
                results[name] = {
                    'model': model,
                    'train_score': model.score(X_train, y_train)
                }
            except Exception as e:
                print(f"Hata ({name}): {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, name, X):
        """Tahmin yap"""
        if name not in self.trained_models:
            raise ValueError(f"Model henüz eğitilmedi: {name}")
        
        return self.trained_models[name].predict(X)
    
    def predict_proba(self, name, X):
        """Olasılık tahmini yap"""
        if name not in self.trained_models:
            raise ValueError(f"Model henüz eğitilmedi: {name}")
        
        return self.trained_models[name].predict_proba(X)
    
    def cross_validate(self, name, X, y, cv=5):
        """K-Fold Cross Validation uygula"""
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
        
        print(f"\n{name} için {cv}-Fold Cross Validation...")
        
        scores = cross_val_score(
            self.models[name], X, y, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"CV Skorları: {scores}")
        print(f"Ortalama: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def cross_validate_all(self, X, y, cv=5):
        """Tüm modeller için cross validation"""
        all_results = {}
        
        for name in self.models:
            try:
                all_results[name] = self.cross_validate(name, X, y, cv)
            except Exception as e:
                print(f"Hata ({name}): {str(e)}")
                all_results[name] = {'error': str(e)}
        
        return all_results
    
    def get_cv_predictions(self, name, X, y, cv=5):
        """Cross validation tahminlerini al (confusion matrix için)"""
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
        
        y_pred = cross_val_predict(self.models[name], X, y, cv=cv)
        return y_pred
    
    def save_model(self, name, filepath):
        """Modeli kaydet"""
        if name not in self.trained_models:
            raise ValueError(f"Model henüz eğitilmedi: {name}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.trained_models[name], filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, name, filepath):
        """Modeli yükle"""
        self.trained_models[name] = joblib.load(filepath)
        print(f"Model yüklendi: {filepath}")
        return self.trained_models[name]
    
    def save_all_models(self, directory):
        """Tüm eğitilmiş modelleri kaydet"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.trained_models.items():
            safe_name = name.replace(' ', '_').lower()
            filepath = os.path.join(directory, f"{safe_name}.pkl")
            joblib.dump(model, filepath)
            print(f"Kaydedildi: {filepath}")
    
    def get_best_model(self, results):
        """En iyi performans gösteren modeli bul"""
        best_name = None
        best_score = 0
        
        for name, result in results.items():
            if 'error' not in result:
                score = result.get('mean', result.get('train_score', 0))
                if score > best_score:
                    best_score = score
                    best_name = name
        
        return best_name, best_score


def main():
    """Test fonksiyonu"""
    from data_preprocessing import DataPreprocessor
    
    # Veri hazırlığı
    preprocessor = DataPreprocessor()
    data_path = "../data/phishing_dataset.csv"
    
    # Örnek veri oluştur
    df = preprocessor.create_sample_dataset(data_path)
    df = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df, target_column='label')
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, validation_size=0)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Sınıflandırıcıları başlat
    classifiers = PhishingClassifiers()
    
    # Tüm modelleri eğit
    print("\n" + "="*60)
    print("TÜM MODELLERİ EĞİTME")
    print("="*60)
    results = classifiers.train_all_models(X_train_scaled, y_train)
    
    # Cross validation
    print("\n" + "="*60)
    print("CROSS VALIDATION")
    print("="*60)
    cv_results = classifiers.cross_validate_all(X_train_scaled, y_train, cv=5)
    
    # En iyi modeli bul
    best_model, best_score = classifiers.get_best_model(cv_results)
    print(f"\n{'='*60}")
    print(f"EN İYİ MODEL: {best_model} (Skor: {best_score:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
