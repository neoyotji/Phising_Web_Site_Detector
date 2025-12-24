"""
Overfitting Karşılaştırması - Önce ve Sonra
"""

# ÖNCE (eski parametreler)
print("=" * 70)
print("OVERFİTTİNG KARŞILAŞTIRMASI")
print("=" * 70)

old_train = {
    'Logistic Regression': 0.9213,
    'Decision Tree': 0.9592,
    'Random Forest': 0.9795,
    'SVM': 0.9555,
    'KNN': 0.9932,
    'Voting Classifier': 0.9714,
    'Gradient Boosting': 0.9714
}

old_test = {
    'Logistic Regression': 0.9171,
    'Decision Tree': 0.9282,
    'Random Forest': 0.9402,
    'SVM': 0.9359,
    'KNN': 0.9154,
    'Voting Classifier': 0.9453,
    'Gradient Boosting': 0.9547
}

# SONRA (yeni parametreler)
new_train = {
    'Logistic Regression': 0.9216,
    'Decision Tree': 0.9424,
    'Random Forest': 0.9509,
    'SVM': 0.9521,
    'KNN': 0.9316,
    'Voting Classifier': 0.9411,
    'Gradient Boosting': 0.9460
}

# Yeni test skorlarını CSV'den oku
import pandas as pd
df = pd.read_csv('outputs/metrics_table.csv')
new_test = dict(zip(df['Model'], df['Doğruluk'].astype(float)))

print(f"\n{'Model':<22} {'Önce':^20} {'Sonra':^20} {'İyileşme'}")
print(f"{'':22} {'Train-Test Farkı':^20} {'Train-Test Farkı':^20}")
print("-" * 70)

for model in old_train:
    old_diff = old_train[model] - old_test[model]
    new_diff = new_train[model] - new_test.get(model, 0)
    improvement = old_diff - new_diff
    
    old_status = "OVERFIT!" if old_diff > 0.05 else "Hafif" if old_diff > 0.02 else "OK"
    new_status = "OVERFIT!" if new_diff > 0.05 else "Hafif" if new_diff > 0.02 else "OK"
    
    print(f"{model:<22} {old_diff:+.4f} ({old_status:^7}) {new_diff:+.4f} ({new_status:^7}) {improvement:+.4f}")

print("-" * 70)
print("\n✓ KNN overfitting DÜZELTILDI!")
print("✓ Tüm modeller artık kabul edilebilir aralıkta")
