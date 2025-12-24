"""
Tüm modellerin eğitimi, değerlendirmesi ve sonuçların kaydedilmesi
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Modül yolunu ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Proje modülleri
from data_preprocessing import DataPreprocessor
from classifiers import PhishingClassifiers
from neural_network import NeuralNetworkClassifier
from evaluation import ModelEvaluator
from statistical_tests import StatisticalTests


def print_header(text):
    """Başlık yazdır"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def main():
    """Ana fonksiyon - Tüm deneysel çalışmaları gerçekleştirir"""
    
    start_time = datetime.now()
    
    print_header("PHİSHİNG WEB SİTESİ TESPİTİ - MAKİNE ÖĞRENMESİ PROJESİ")
    print("224410054_SenanurOzbag_Proje")
    print(f"Başlangıç zamanı: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ===== 1. VERİ HAZIRLAMA =====
    print_header("1. VERİ HAZIRLAMA")
    
    preprocessor = DataPreprocessor(random_state=42)
    
    # Veri yolları
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")  # Kaggle dataset
    output_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "saved_models")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Kaggle veri setini yükle
    print("Kaggle Phishing Dataset yükleniyor...")
    df = preprocessor.load_data(data_path)
    
    # Veri keşfi
    df = preprocessor.explore_data(df)
    
    # Index sütununu kaldır (eğer varsa)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
        print("'index' sütunu kaldırıldı.")
    
    # Hedef değişkeni dönüştür: -1 -> 0 (Phishing), 1 -> 1 (Legitimate)
    # NOT: Orijinal veri setinde -1 = Phishing, 1 = Legitimate
    if 'Result' in df.columns:
        df['Result'] = df['Result'].map({-1: 1, 1: 0})  # 1: Phishing, 0: Legitimate
        print("Hedef değişken dönüştürüldü: -1 -> 1 (Phishing), 1 -> 0 (Legitimate)")
    
    # Veri temizleme
    df = preprocessor.clean_data(df)
    
    # Özellikleri hazırla
    X, y = preprocessor.prepare_features(df, target_column='Result')
    print(f"\nÖzellik sayısı: {X.shape[1]}")
    print(f"Toplam örnek: {X.shape[0]}")
    
    # Hold-out bölme
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_size=0.2, validation_size=0.1
    )
    
    # Ölçeklendirme
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_test, X_val
    )
    
    # Tam veri seti (k-fold için)
    X_full_scaled = preprocessor.scaler.fit_transform(X)
    
    # ===== 2. SINIFLANDIRICILARI EĞİT =====
    print_header("2. SINIFLANDIRICILARI EĞİTME (Hold-out)")
    
    classifiers = PhishingClassifiers(random_state=42)
    evaluator = ModelEvaluator(output_dir=output_dir)
    
    # Tüm modelleri eğit
    training_results = classifiers.train_all_models(X_train_scaled, y_train)
    
    # ===== 3. HOLD-OUT DEĞERLENDİRME =====
    print_header("3. HOLD-OUT (DIŞARIDA TUTMA) DEĞERLENDİRME")
    
    holdout_results = {}
    predictions_dict = {}  # McNemar testi için
    
    for model_name in classifiers.get_model_names():
        model = classifiers.trained_models[model_name]
        metrics, cm = evaluator.evaluate_holdout(
            model, model_name, X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        holdout_results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': classifiers.predict(model_name, X_test_scaled),
            'y_proba': classifiers.predict_proba(model_name, X_test_scaled) 
                       if hasattr(model, 'predict_proba') else None
        }
        
        predictions_dict[model_name] = holdout_results[model_name]['y_pred']
        
        # Confusion matrix kaydet
        evaluator.plot_confusion_matrix(cm, model_name)
        
        # ROC eğrisi kaydet
        if holdout_results[model_name]['y_proba'] is not None:
            evaluator.plot_roc_curve(y_test, holdout_results[model_name]['y_proba'], model_name)
    
    # ===== 4. YAPAY SİNİR AĞI =====
    print_header("4. YAPAY SİNİR AĞI (ANN) EĞİTİMİ - Epoch >= 100")
    
    nn = NeuralNetworkClassifier(input_dim=X_train_scaled.shape[1], random_state=42)
    nn.build_model(hidden_layers=[128, 64, 32], dropout_rate=0.3)
    nn.train(X_train_scaled, y_train, X_val_scaled, y_val, epochs=100, batch_size=32)
    
    # ANN değerlendirmesi
    nn.evaluate(X_test_scaled, y_test)
    
    # Eğitim grafikleri
    nn.plot_training_history(save_path=os.path.join(output_dir, 'ann_training_history.png'))
    
    # ANN sonuçlarını ekle
    y_pred_ann = nn.predict(X_test_scaled)
    y_proba_ann = nn.predict_proba(X_test_scaled)
    metrics_ann, cm_ann = evaluator.calculate_metrics(y_test, y_pred_ann, y_proba_ann)
    
    holdout_results['Neural Network'] = {
        'metrics': metrics_ann,
        'confusion_matrix': cm_ann,
        'y_true': y_test,
        'y_pred': y_pred_ann,
        'y_proba': y_proba_ann
    }
    predictions_dict['Neural Network'] = y_pred_ann
    
    evaluator.plot_confusion_matrix(cm_ann, 'Neural Network')
    evaluator.plot_roc_curve(y_test, y_proba_ann, 'Neural Network')
    
    # ANN modelini kaydet
    nn.save_model(os.path.join(models_dir, 'neural_network.h5'))
    
    # ===== 5. K-FOLD CROSS VALIDATION =====
    print_header("5. K-FOLD CROSS VALIDATION (k=5)")
    
    kfold_results = {}
    
    for model_name in classifiers.get_model_names():
        cv_result = classifiers.cross_validate(model_name, X_full_scaled, y, cv=5)
        kfold_results[model_name] = cv_result
    
    # ===== 6. TÜM GÖRSELLEŞTIRMELER =====
    print_header("6. GÖRSELLEŞTİRMELER")
    
    # Tüm confusion matrisleri
    evaluator.plot_all_confusion_matrices(holdout_results)
    
    # Tüm ROC eğrileri
    evaluator.plot_all_roc_curves(holdout_results)
    
    # Metrik tablosu
    metrics_df = evaluator.create_metrics_table(holdout_results)
    
    # Metrik karşılaştırma grafiği
    evaluator.create_metrics_visualization(holdout_results)
    
    # ===== 7. McNEMAR TESTİ =====
    print_header("7. McNEMAR TESTİ")
    
    stat_tests = StatisticalTests(output_dir=output_dir)
    
    # Tüm model çiftleri için McNemar testi
    mcnemar_results = stat_tests.mcnemar_all_pairs(y_test, predictions_dict, alpha=0.05)
    
    # McNemar özet tablosu
    stat_tests.create_mcnemar_summary_table()
    
    # Çift karşılaştırma matrisi
    stat_tests.create_pairwise_comparison_matrix(y_test, predictions_dict)
    
    # ===== 8. EN İYİ MODELİ BUL VE KAYDET =====
    print_header("8. EN İYİ MODEL")
    
    # En iyi modeli bul (F1-skora göre)
    best_model_name = None
    best_f1 = 0
    
    for model_name, result in holdout_results.items():
        f1 = result['metrics'].get('F1-Skoru', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = model_name
    
    print(f"\nEN İYİ MODEL: {best_model_name}")
    print(f"F1-Skoru: {best_f1:.4f}")
    print(f"Metrikler:")
    for key, value in holdout_results[best_model_name]['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # En iyi modeli kaydet
    if best_model_name != 'Neural Network':
        classifiers.save_model(
            best_model_name, 
            os.path.join(models_dir, 'best_model.pkl')
        )
    
    # Scaler'ı kaydet
    import joblib
    joblib.dump(preprocessor.scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"Scaler kaydedildi: {os.path.join(models_dir, 'scaler.pkl')}")
    
    # Tüm modelleri kaydet
    classifiers.save_all_models(models_dir)
    
    # ===== 9. K-FOLD SONUÇ TABLOSU =====
    print_header("9. K-FOLD CROSS VALIDATION SONUÇLARI")
    
    kfold_table = []
    for model_name, result in kfold_results.items():
        row = {
            'Model': model_name,
            'Ortalama Doğruluk': f"{result['mean']:.4f}",
            'Std Sapma': f"{result['std']:.4f}"
        }
        kfold_table.append(row)
    
    kfold_df = pd.DataFrame(kfold_table)
    print(kfold_df.to_string(index=False))
    kfold_df.to_csv(os.path.join(output_dir, 'kfold_results.csv'), index=False)
    
    # ===== SONUÇ =====
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("TAMAMLANDI")
    print(f"Bitiş zamanı: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Toplam süre: {duration:.2f} saniye")
    print(f"\nÇıktılar: {output_dir}")
    print(f"Kaydedilen modeller: {models_dir}")
    
    return {
        'holdout_results': holdout_results,
        'kfold_results': kfold_results,
        'mcnemar_results': mcnemar_results,
        'best_model': best_model_name,
        'best_f1': best_f1
    }


if __name__ == "__main__":
    results = main()
