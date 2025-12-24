"""
McNemar Testi ve model karşılaştırma istatistikleri
"""

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
import itertools
import os


class StatisticalTests:
    """İstatistiksel test sınıfı"""
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.results = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2, model1_name, model2_name, alpha=0.05):
        """
        McNemar Testi
        
        İki sınıflandırıcının performanslarının istatistiksel olarak anlamlı 
        şekilde farklı olup olmadığını test eder.
        
        H0: İki model aynı performansa sahiptir
        H1: İki model farklı performansa sahiptir
        """
        
        print(f"\n{'='*60}")
        print(f"McNEMAR TESTİ: {model1_name} vs {model2_name}")
        print(f"{'='*60}")
        
        # Kontenjans tablosu oluştur
        # Model1 doğru, Model2 doğru (a)
        # Model1 doğru, Model2 yanlış (b)
        # Model1 yanlış, Model2 doğru (c)
        # Model1 yanlış, Model2 yanlış (d)
        
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        a = np.sum(correct1 & correct2)      # İkisi de doğru
        b = np.sum(correct1 & ~correct2)     # Sadece Model1 doğru
        c = np.sum(~correct1 & correct2)     # Sadece Model2 doğru
        d = np.sum(~correct1 & ~correct2)    # İkisi de yanlış
        
        contingency_table = np.array([[a, b], [c, d]])
        
        print(f"\nKontenjans Tablosu:")
        print(f"                    {model2_name}")
        print(f"                    Doğru   Yanlış")
        print(f"{model1_name:12} Doğru    {a:5d}    {b:5d}")
        print(f"             Yanlış   {c:5d}    {d:5d}")
        
        # McNemar testi
        # b ve c değerleri önemli (uyumsuz çiftler)
        
        if b + c < 25:
            # Küçük örneklem - exact binomial test kullan
            print("\nKüçük örneklem - Exact McNemar Testi kullanılıyor")
            result = mcnemar(contingency_table, exact=True)
        else:
            # Büyük örneklem - chi-square approximation
            print("\nBüyük örneklem - Chi-square yaklaşımı kullanılıyor")
            result = mcnemar(contingency_table, exact=False, correction=True)
        
        statistic = result.statistic
        p_value = result.pvalue
        
        print(f"\nTest İstatistiği: {statistic:.4f}")
        print(f"P-değeri: {p_value:.6f}")
        print(f"Anlamlılık düzeyi (α): {alpha}")
        
        # Karar
        if p_value < alpha:
            decision = "FARK İSTATİSTİKSEL OLARAK ANLAMLI"
            significant = True
            print(f"\n✓ {decision}")
            print(f"  H0 reddedildi: İki model farklı performansa sahiptir.")
            
            # Hangi model daha iyi?
            accuracy1 = np.mean(correct1)
            accuracy2 = np.mean(correct2)
            
            if accuracy1 > accuracy2:
                print(f"  {model1_name} daha iyi performans gösteriyor ({accuracy1:.4f} > {accuracy2:.4f})")
            else:
                print(f"  {model2_name} daha iyi performans gösteriyor ({accuracy2:.4f} > {accuracy1:.4f})")
        else:
            decision = "FARK İSTATİSTİKSEL OLARAK ANLAMLI DEĞİL"
            significant = False
            print(f"\n✗ {decision}")
            print(f"  H0 reddedilemedi: İki model benzer performansa sahiptir.")
        
        result_dict = {
            'model1': model1_name,
            'model2': model2_name,
            'contingency_table': contingency_table,
            'a': a, 'b': b, 'c': c, 'd': d,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': significant,
            'decision': decision
        }
        
        key = f"{model1_name}_vs_{model2_name}"
        self.results[key] = result_dict
        
        return result_dict
    
    def mcnemar_all_pairs(self, y_true, predictions_dict, alpha=0.05):
        """Tüm model çiftleri için McNemar testi uygula"""
        
        print("\n" + "="*70)
        print("TÜM MODEL ÇİFTLERİ İÇİN McNEMAR TESTİ")
        print("="*70)
        
        model_names = list(predictions_dict.keys())
        pairs = list(itertools.combinations(model_names, 2))
        
        all_results = []
        
        for model1_name, model2_name in pairs:
            y_pred1 = predictions_dict[model1_name]
            y_pred2 = predictions_dict[model2_name]
            
            result = self.mcnemar_test(
                y_true, y_pred1, y_pred2, 
                model1_name, model2_name, alpha
            )
            all_results.append(result)
        
        return all_results
    
    def create_mcnemar_summary_table(self, results=None, save=True):
        """McNemar test sonuçlarının özet tablosu"""
        
        if results is None:
            results = self.results
        
        table_data = []
        
        for key, result in results.items():
            row = {
                'Model 1': result['model1'],
                'Model 2': result['model2'],
                'b (M1✓, M2✗)': result['b'],
                'c (M1✗, M2✓)': result['c'],
                'Chi-square': f"{result['statistic']:.4f}",
                'P-değeri': f"{result['p_value']:.6f}",
                'Anlamlı mı?': 'Evet' if result['significant'] else 'Hayır'
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        print("\n" + "="*90)
        print("McNEMAR TESTİ ÖZET TABLOSU")
        print("="*90)
        print(df.to_string(index=False))
        
        if save:
            filepath = os.path.join(self.output_dir, 'mcnemar_results.csv')
            df.to_csv(filepath, index=False)
            print(f"\nTablo kaydedildi: {filepath}")
        
        return df
    
    def create_pairwise_comparison_matrix(self, y_true, predictions_dict, save=True):
        """Çift karşılaştırma matrisi (p-değerleri)"""
        
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        p_matrix = np.ones((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    y_pred1 = predictions_dict[model1]
                    y_pred2 = predictions_dict[model2]
                    
                    # McNemar testi
                    correct1 = (y_pred1 == y_true)
                    correct2 = (y_pred2 == y_true)
                    
                    a = np.sum(correct1 & correct2)
                    b = np.sum(correct1 & ~correct2)
                    c = np.sum(~correct1 & correct2)
                    d = np.sum(~correct1 & ~correct2)
                    
                    contingency_table = np.array([[a, b], [c, d]])
                    
                    try:
                        if b + c < 25:
                            result = mcnemar(contingency_table, exact=True)
                        else:
                            result = mcnemar(contingency_table, exact=False, correction=True)
                        p_matrix[i, j] = result.pvalue
                        p_matrix[j, i] = result.pvalue
                    except:
                        p_matrix[i, j] = 1.0
                        p_matrix[j, i] = 1.0
        
        df = pd.DataFrame(p_matrix, index=model_names, columns=model_names)
        
        print("\n" + "="*80)
        print("ÇİFT KARŞILAŞTIRMA MATRİSİ (P-değerleri)")
        print("="*80)
        print(df.round(4).to_string())
        
        if save:
            filepath = os.path.join(self.output_dir, 'pairwise_comparison_matrix.csv')
            df.to_csv(filepath)
            print(f"\nMatris kaydedildi: {filepath}")
        
        return df
    
    def bonferroni_correction(self, p_values, alpha=0.05):
        """Bonferroni düzeltmesi uygula"""
        
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        
        print(f"\nBonferroni Düzeltmesi:")
        print(f"  Test sayısı: {n_tests}")
        print(f"  Orijinal α: {alpha}")
        print(f"  Düzeltilmiş α: {corrected_alpha:.6f}")
        
        significant = [p < corrected_alpha for p in p_values]
        
        return corrected_alpha, significant


def main():
    """Test fonksiyonu"""
    print("İstatistiksel testler modülü yüklendi.")
    
    # Örnek test
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.randint(0, 2, n)
    y_pred1 = y_true.copy()
    y_pred1[:100] = 1 - y_pred1[:100]  # %10 hata
    
    y_pred2 = y_true.copy()
    y_pred2[:150] = 1 - y_pred2[:150]  # %15 hata
    
    tests = StatisticalTests()
    tests.mcnemar_test(y_true, y_pred1, y_pred2, "Model A", "Model B")


if __name__ == "__main__":
    main()
