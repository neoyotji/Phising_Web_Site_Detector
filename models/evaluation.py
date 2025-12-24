"""
Karışıklık matrisi, ROC eğrileri, metrikler ve görselleştirmeler
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import pandas as pd
import os


class ModelEvaluator:
    """Model değerlendirme sınıfı"""
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Tüm metrikleri hesapla"""
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Metrikler
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)  # TPR = Recall = Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        
        metrics = {
            'Doğruluk (Accuracy)': accuracy,
            'Duyarlılık (Sensitivity/Recall)': sensitivity,
            'Özgüllük (Specificity)': specificity,
            'Kesinlik (Precision)': precision,
            'F1-Skoru': f1,
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        }
        
        # AUC (eğer olasılık varsa)
        if y_proba is not None:
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        
        return metrics, cm
    
    def evaluate_holdout(self, model, model_name, X_train, X_test, y_train, y_test):
        """Hold-out (Dışarıda Tutma) validasyonu"""
        
        print(f"\n{'='*60}")
        print(f"HOLD-OUT DEĞERLENDİRME: {model_name}")
        print(f"{'='*60}")
        
        # Tahmin
        y_pred = model.predict(X_test)
        
        # Olasılık tahmini
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        # Metrikler
        metrics, cm = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Sonuçları yazdır
        print("\nMetrikler:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Sonuçları kaydet
        self.results[f"{model_name}_holdout"] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics, cm
    
    def evaluate_kfold(self, model, model_name, X, y, cv=5):
        """K-Fold Cross Validation değerlendirmesi"""
        
        print(f"\n{'='*60}")
        print(f"{cv}-FOLD CROSS VALIDATION: {model_name}")
        print(f"{'='*60}")
        
        # Cross validation tahminleri
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        all_metrics = []
        fold_cms = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Modeli eğit ve tahmin et
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            
            # Olasılık
            y_proba_fold = None
            if hasattr(model, 'predict_proba'):
                y_proba_fold = model.predict_proba(X_test_fold)
            
            # Metrikler
            fold_metrics, fold_cm = self.calculate_metrics(y_test_fold, y_pred_fold, y_proba_fold)
            all_metrics.append(fold_metrics)
            fold_cms.append(fold_cm)
            
            print(f"Fold {fold}: Accuracy={fold_metrics['Doğruluk (Accuracy)']:.4f}, F1={fold_metrics['F1-Skoru']:.4f}")
        
        # Ortalama metrikler
        avg_metrics = {}
        std_metrics = {}
        
        metric_keys = ['Doğruluk (Accuracy)', 'Duyarlılık (Sensitivity/Recall)', 
                      'Özgüllük (Specificity)', 'Kesinlik (Precision)', 'F1-Skoru']
        
        if 'AUC' in all_metrics[0]:
            metric_keys.append('AUC')
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        
        print(f"\nOrtalama Sonuçlar ({cv}-Fold):")
        for key in metric_keys:
            print(f"  {key}: {avg_metrics[key]:.4f} (+/- {std_metrics[key]:.4f})")
        
        # Toplam confusion matrix
        total_cm = np.sum(fold_cms, axis=0)
        
        self.results[f"{model_name}_kfold"] = {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_fold_metrics': all_metrics,
            'confusion_matrix': total_cm,
            'cv': cv
        }
        
        return avg_metrics, std_metrics, total_cm
    
    def plot_confusion_matrix(self, cm, model_name, save=True):
        """Karışıklık matrisini görselleştir"""
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'Karışıklık Matrisi - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Tahmin Edilen', fontsize=12)
        plt.ylabel('Gerçek', fontsize=12)
        plt.tight_layout()
        
        if save:
            safe_name = model_name.replace(' ', '_').lower()
            filepath = os.path.join(self.output_dir, f'confusion_matrix_{safe_name}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Kaydedildi: {filepath}")
        
        plt.close()
    
    def plot_all_confusion_matrices(self, results_dict, save=True):
        """Tüm modellerin karışıklık matrislerini tek figürde çiz"""
        
        n_models = len(results_dict)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(results_dict.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Phishing'],
                       yticklabels=['Legitimate', 'Phishing'])
            axes[idx].set_title(model_name, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Tahmin', fontsize=10)
            axes[idx].set_ylabel('Gerçek', fontsize=10)
        
        # Boş eksenleri gizle
        for idx in range(len(results_dict), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Tüm Modellerin Karışıklık Matrisleri', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'all_confusion_matrices.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Kaydedildi: {filepath}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_proba, model_name, save=True):
        """ROC eğrisini çiz"""
        
        if y_proba is None:
            print(f"ROC eğrisi çizilemedi - {model_name}: Olasılık değerleri yok")
            return
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Eğrisi (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
        plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
        plt.title(f'ROC Eğrisi - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            safe_name = model_name.replace(' ', '_').lower()
            filepath = os.path.join(self.output_dir, f'roc_curve_{safe_name}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Kaydedildi: {filepath}")
        
        plt.close()
        
        return fpr, tpr, roc_auc
    
    def plot_all_roc_curves(self, results_dict, save=True):
        """Tüm ROC eğrilerini tek figürde çiz"""
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        for (model_name, result), color in zip(results_dict.items(), colors):
            y_proba = result.get('y_proba')
            y_true = result.get('y_true')
            
            if y_proba is None or y_true is None:
                continue
            
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Rastgele')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı (FPR)', fontsize=12)
        plt.ylabel('Doğru Pozitif Oranı (TPR)', fontsize=12)
        plt.title('Tüm Modellerin ROC Eğrileri', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'all_roc_curves.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Kaydedildi: {filepath}")
        
        plt.close()
    
    def create_metrics_table(self, results_dict, save=True):
        """Metrik sonuçlarını tablo olarak oluştur"""
        
        table_data = []
        
        for model_name, result in results_dict.items():
            metrics = result.get('metrics', result.get('avg_metrics', {}))
            
            row = {
                'Model': model_name.replace('_holdout', '').replace('_kfold', ''),
                'Doğruluk': f"{metrics.get('Doğruluk (Accuracy)', 0):.4f}",
                'Duyarlılık': f"{metrics.get('Duyarlılık (Sensitivity/Recall)', 0):.4f}",
                'Özgüllük': f"{metrics.get('Özgüllük (Specificity)', 0):.4f}",
                'Kesinlik': f"{metrics.get('Kesinlik (Precision)', 0):.4f}",
                'F1-Skoru': f"{metrics.get('F1-Skoru', 0):.4f}",
            }
            
            if 'AUC' in metrics:
                row['AUC'] = f"{metrics.get('AUC', 0):.4f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANS TABLOSU")
        print("="*80)
        print(df.to_string(index=False))
        
        if save:
            filepath = os.path.join(self.output_dir, 'metrics_table.csv')
            df.to_csv(filepath, index=False)
            print(f"\nTablo kaydedildi: {filepath}")
        
        return df
    
    def create_metrics_visualization(self, results_dict, save=True):
        """Metrik karşılaştırma bar grafiği"""
        
        models = []
        metrics_data = {'Doğruluk': [], 'Duyarlılık': [], 'Özgüllük': [], 'F1-Skoru': []}
        
        for model_name, result in results_dict.items():
            metrics = result.get('metrics', result.get('avg_metrics', {}))
            clean_name = model_name.replace('_holdout', '').replace('_kfold', '')
            models.append(clean_name)
            
            metrics_data['Doğruluk'].append(metrics.get('Doğruluk (Accuracy)', 0))
            metrics_data['Duyarlılık'].append(metrics.get('Duyarlılık (Sensitivity/Recall)', 0))
            metrics_data['Özgüllük'].append(metrics.get('Özgüllük (Specificity)', 0))
            metrics_data['F1-Skoru'].append(metrics.get('F1-Skoru', 0))
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - 1.5*width, metrics_data['Doğruluk'], width, label='Doğruluk', color='#2ecc71')
        bars2 = ax.bar(x - 0.5*width, metrics_data['Duyarlılık'], width, label='Duyarlılık', color='#3498db')
        bars3 = ax.bar(x + 0.5*width, metrics_data['Özgüllük'], width, label='Özgüllük', color='#e74c3c')
        bars4 = ax.bar(x + 1.5*width, metrics_data['F1-Skoru'], width, label='F1-Skoru', color='#9b59b6')
        
        ax.set_ylabel('Skor', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Kaydedildi: {filepath}")
        
        plt.close()


def main():
    """Test fonksiyonu"""
    print("Değerlendirme modülü yüklendi.")


if __name__ == "__main__":
    main()
