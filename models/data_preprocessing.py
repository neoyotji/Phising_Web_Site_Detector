"""
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class DataPreprocessor:
    """Veri ön işleme sınıfı"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self, filepath):
        """Veri setini yükle"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Desteklenmeyen dosya formatı. CSV veya XLSX kullanın.")
        
        print(f"Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        return df
    
    def explore_data(self, df):
        """Veri keşfi"""
        print("\n" + "="*50)
        print("VERİ KEŞFİ")
        print("="*50)
        print(f"\nVeri boyutu: {df.shape}")
        print(f"\nSütun tipleri:\n{df.dtypes}")
        print(f"\nEksik değerler:\n{df.isnull().sum()}")
        print(f"\nHedef değişken dağılımı:\n{df.iloc[:, -1].value_counts()}")
        print(f"\nİstatistiksel özet:\n{df.describe()}")
        return df
    
    def clean_data(self, df):
        """Veri temizleme"""
        # Eksik değerleri doldur
        df = df.fillna(df.median(numeric_only=True))
        
        # Sonsuz değerleri temizle
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Duplicate satırları kaldır
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            print(f"{removed_rows} duplicate satır kaldırıldı.")
        
        return df
    
    def prepare_features(self, df, target_column=None):
        """Özellikleri hazırla"""
        if target_column is None:
            # Son sütun hedef değişken
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        
        # Kategorik değişkenleri encode et
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Hedef değişkeni encode et
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.values
        
        self.feature_names = X.columns.tolist()
        
        return X.values, y
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """Hold-out yöntemi ile veri bölme"""
        # Train ve Test ayrımı
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Validation seti oluştur (opsiyonel)
        if validation_size > 0:
            val_ratio = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_ratio, 
                random_state=self.random_state, stratify=y_train
            )
            print(f"Veri bölündü - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        print(f"Veri bölündü - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, X_val=None):
        """Özellikleri ölçeklendir"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            return X_train_scaled, X_val_scaled, X_test_scaled
        
        return X_train_scaled, X_test_scaled
    
    def get_kfold_splits(self, X, y, n_splits=5):
        """K-Fold Cross Validation için split'ler oluştur"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return list(skf.split(X, y))
    
    def create_sample_dataset(self, output_path, n_samples=10000):
        """Örnek phishing veri seti oluştur (eğer veri seti yoksa)"""
        np.random.seed(self.random_state)
        
        # Phishing ve legitimate URL'ler için özellikler
        features = {
            'url_length': np.random.randint(10, 200, n_samples),
            'has_https': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'has_ip_address': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'num_dots': np.random.randint(1, 10, n_samples),
            'num_hyphens': np.random.randint(0, 5, n_samples),
            'num_underscores': np.random.randint(0, 3, n_samples),
            'num_slashes': np.random.randint(1, 8, n_samples),
            'num_questionmarks': np.random.randint(0, 3, n_samples),
            'num_equals': np.random.randint(0, 5, n_samples),
            'num_at_symbols': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'num_ampersands': np.random.randint(0, 4, n_samples),
            'num_digits': np.random.randint(0, 20, n_samples),
            'domain_length': np.random.randint(3, 50, n_samples),
            'path_length': np.random.randint(0, 100, n_samples),
            'subdomain_count': np.random.randint(0, 5, n_samples),
            'has_suspicious_words': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'domain_age_days': np.random.randint(0, 3650, n_samples),
            'alexa_rank': np.random.randint(0, 1000000, n_samples),
            'google_index': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'page_rank': np.random.randint(0, 10, n_samples),
            'dns_record': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'web_traffic': np.random.randint(0, 100000, n_samples),
            'links_in_tags': np.random.randint(0, 50, n_samples),
            'links_pointing_to_page': np.random.randint(0, 500, n_samples),
            'statistical_report': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        }
        
        df = pd.DataFrame(features)
        
        # Phishing olasılığını hesapla (bazı özelliklere göre)
        phishing_score = (
            (df['url_length'] > 75).astype(int) * 2 +
            (df['has_https'] == 0).astype(int) * 2 +
            (df['has_ip_address'] == 1).astype(int) * 3 +
            (df['num_dots'] > 4).astype(int) +
            (df['has_suspicious_words'] == 1).astype(int) * 2 +
            (df['domain_age_days'] < 180).astype(int) * 2 +
            (df['alexa_rank'] > 500000).astype(int) +
            (df['google_index'] == 0).astype(int) * 2 +
            (df['dns_record'] == 0).astype(int) * 2 +
            np.random.normal(0, 2, n_samples)
        )
        
        # Threshold ile sınıflandır
        df['label'] = (phishing_score > 5).astype(int)  # 1: Phishing, 0: Legitimate
        
        # Dengeleme
        phishing_count = df['label'].sum()
        legitimate_count = len(df) - phishing_count
        
        print(f"Phishing: {phishing_count}, Legitimate: {legitimate_count}")
        
        # Veri setini kaydet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Örnek veri seti oluşturuldu: {output_path}")
        
        return df


def main():
    """Test fonksiyonu"""
    preprocessor = DataPreprocessor()
    
    # Örnek veri seti oluştur
    data_path = "data/phishing_dataset.csv"
    df = preprocessor.create_sample_dataset(data_path)
    
    # Veri keşfi
    df = preprocessor.explore_data(df)
    
    # Veri temizleme
    df = preprocessor.clean_data(df)
    
    # Özellikleri hazırla
    X, y = preprocessor.prepare_features(df, target_column='label')
    
    # Veriyi böl
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Ölçeklendir
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test, X_val)
    
    print(f"\nÖlçeklendirilmiş veri boyutları:")
    print(f"Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")


if __name__ == "__main__":
    main()
