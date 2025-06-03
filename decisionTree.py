from google.colab import files
uploaded = files.upload()

import pandas as pd

df = pd.read_csv("Wholesale customers data.csv")  # dosya adını tam gir!
df.head()  # İlk 5 satırı göster

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Özellik ve hedefi ayır
X = df.drop("Channel", axis=1)
y = df["Channel"]

# Eğitimi ve testi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini eğit
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Ağacı çiz
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["Horeca", "Retail"], filled=True)
plt.show()

print(df.columns.tolist())

X = df.drop("Channel", axis=1)
y = df["Channel"]

import pandas as pd

# Doğru ayırıcı ile oku
df = pd.read_csv("Wholesale customers data.csv", sep=";")

# Sütun isimlerini kontrol et
print(df.columns.tolist())

X = df.drop("Channel", axis=1)
y = df["Channel"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi eğitim ve test olarak ayır (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modelini oluştur ve eğit
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Tahmin yap ve doğruluğu hesapla
y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["1", "2"], filled=True)
plt.show()

model = DecisionTreeClassifier(max_depth=3)

# Modeli eğit
model.fit(X_train, y_train)

# Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# Başarı oranını yazdır
from sklearn.metrics import accuracy_score
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns,
          class_names=[str(cls) for cls in model.classes_],
          filled=True)
plt.show()

model = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)

# Modeli eğit
model.fit(X_train, y_train)

# Test verileriyle tahmin yap
y_pred = model.predict(X_test)

from sklearn.model_selection import train_test_split

# Özellikler (X) ve etiketler (y) daha önce tanımlanmış olmalı:
# Örnek olarak:
# X = df.drop("Channel", axis=1)
# y = df["Channel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd

# CSV dosyasını uygun ayırıcıyla oku
df = pd.read_csv("Wholesale customers data.csv", sep=";")

# Özellikler ve hedef sütunu ayır
X = df.drop("Channel", axis=1)
y = df["Channel"]

from google.colab import files
uploaded= files.upload()

from sklearn.model_selection import train_test_split

# Özellikler (X) ve etiketler (y) daha önce tanımlanmış olmalı:
# Örnek olarak:
# X = df.drop("Channel", axis=1)
# y = df["Channel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = df.drop("Channel", axis=1)
y = df["Channel"]

import pandas as pd

# CSV dosyasını uygun ayırıcıyla oku
df = pd.read_csv("Wholesale customers data.csv", sep=";")

# Özellikler ve hedef sütunu ayır
X = df.drop("Channel", axis=1)
y = df["Channel"]

from sklearn.model_selection import train_test_split

# Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

model = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Korelasyon matrisini hesapla
corr_matrix = df.corr()

# Örneğin 'Detergents_Paper' ile diğer özniteliklerin korelasyonunu görselleştirelim
feature = 'Detergents_Paper'
correlations = corr_matrix[feature].drop(feature)  # Kendisiyle olan korelasyonu çıkar

# Çubuk grafiği çiz
plt.figure(figsize=(8,5))
correlations.sort_values().plot(kind='barh', color='skyblue')
plt.title(f"'{feature}' Özelliğinin Diğer Özelliklerle Korelasyonları")
plt.xlabel("Korelasyon Değeri")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import pandas as pd

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("Özellik Önemleri:")
print(feature_importances)

import matplotlib.pyplot as plt
import seaborn as sns

# Sınıf dağılımını hesapla
class_counts = df['Channel'].value_counts().sort_index()

# Matplotlib ile bar grafik
plt.figure(figsize=(6,4))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['skyblue', 'salmon'])
plt.title('Channel Sınıf Dağılımı')
plt.xlabel('Channel')
plt.ylabel('Örnek Sayısı')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Channel', data=df, palette='Set2')
plt.title('Channel Sınıf Dağılımı')
plt.show()

features = df.columns.drop('Channel')

plt.figure(figsize=(16,10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)  # 3x3 grid varsayımı, özellik sayısına göre ayarla
    sns.violinplot(x='Channel', y=feature, data=df, palette='Set2')
    plt.title(f'{feature} Dağılımı')
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Tahminleri yap
y_pred = model.predict(X_test)

# Doğruluk (Accuracy)
accuracy = accuracy_score(y_test, y_pred)

# F1-Score
f1 = f1_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# AUC Skoru (probabilistic output gerekli)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

# Confusion Matrix (TP, TN, FP, FN görmek için)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Sonuçları yazdır
print("Doğruluk (Accuracy):", round(accuracy, 4))
print("F1-Score:", round(f1, 4))
print("Recall:", round(recall, 4))
print("Precision:", round(precision, 4))
print("AUC:", round(auc, 4))
print("Confusion Matrix: TN =", tn, ", FP =", fp, ", FN =", fn, ", TP =", tp)

from sklearn.metrics import accuracy_score

# Eğitim verisiyle tahmin yap
y_train_pred = model.predict(X_train)

# Eğitim doğruluk skorunu hesapla
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", round(train_accuracy, 4))