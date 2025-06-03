# 🛒 Wholesale Customers Veri Seti ile Müşteri Sınıflandırması (Karar Ağacı)

Bu projede, **Wholesale Customers** veri seti kullanılarak müşterilerin alışveriş alışkanlıklarına göre hangi kanalda (Horeca veya Perakende) yer aldıkları **Karar Ağacı (Decision Tree)** algoritması ile tahmin edilmiştir.

---

## 📌 Proje Açıklaması

Bu çalışma, müşterilerin yıllık harcama kalıplarına göre sınıflandırılmasını amaçlamaktadır. Python dili ve **scikit-learn** kütüphanesi kullanılarak karar ağacı modeli kurulmuş, model eğitilmiş ve farklı metriklerle değerlendirilmiştir.

---

## 📊 Veri Seti Bilgileri

- **Kaynak**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/292/wholesale+customers)  
- **Gözlem Sayısı**: 440  
- **Özellik Sayısı**: 8 (sayısal)  
- **Hedef Değişken**: `Channel` (1 = Horeca, 2 = Retail)

### Özellikler:

| Sütun Adı         | Açıklama                                      |
|------------------|-----------------------------------------------|
| Fresh            | Taze ürünler yıllık harcama tutarı            |
| Milk             | Süt ürünleri yıllık harcama tutarı            |
| Grocery          | Market ürünleri yıllık harcama tutarı         |
| Frozen           | Dondurulmuş ürünler yıllık harcama tutarı     |
| Detergents_Paper | Temizlik ürünleri harcama tutarı              |
| Delicassen       | Şarküteri ürünleri harcama tutarı             |
| Region           | Bölge (1: Lizbon, 2: Porto, 3: Diğer)         |
| Channel          | Müşteri kanalı (1: Horeca, 2: Perakende)      |

---

## 🧪 Proje Adımları

1. 📥 Verinin yüklenmesi ve incelenmesi  
2. 🧹 Özellik ve hedef değişken ayrımı  
3. 🧠 Karar ağacı modeli oluşturma ve eğitme  
4. 📊 Performans metriklerinin hesaplanması  
5. 🌲 Modelin görselleştirilmesi  
6. 📈 Özellik önemlerinin ve sınıf dağılımlarının analizi

---
## 🧰 Proje Sunum Videosu 
- **Link**: [Sunum](https://youtu.be/r4IkMQvrd-E?si=E2ZdmZWM1YtFfXLH)
--- 

## ✅ Model Başarım Sonuçları

| Metrik           | Değer     |
|------------------|-----------|
| Doğruluk (Accuracy) | %88.64  |
| F1-Score         | 0.9206    |
| Recall           | 0.8923    |
| Precision        | 0.9508    |
| AUC              | %92.61    |
| Eğitim Başarımı  | %98.01    |

---

## 📉 Görselleştirmeler

- Karar Ağacı Yapısı (plot_tree)
  ![Karar Ağacı](images/egitilmiskarar.png)
- Sınıflara Göre Öznitelik Dağılımları (violin plot)
   ![Violin Plot](images/barplot.png)  
- Sınıf Dağılımı (bar chart)
  
   ![Sınıf Dağılımı](images/sınıfdagilimi.png)
  
- Korelasyon Grafikleri
  
   ![Korelasyon](images/korelasyon.png)

---

## 📚 Diğer Çalışmalarla Karşılaştırma

Bu proje, [Kaggle üzerinde yayınlanan](https://www.kaggle.com/code/sahistapatel96/wholesale-customer-segmentation) bir çalışmadaki KNN, SVM ve Naive Bayes modelleri ile karşılaştırılmıştır.  

KNN modeli daha yüksek doğruluk skoru almış olsa da, **F1-Score**, **Precision** ve **AUC** gibi metriklerde bu projedeki **Decision Tree** modeli daha iyi performans göstermiştir.

---

## 🧰 Kullanılan Teknolojiler

- Python 3  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  
- Google Colab  

---

## 📄 Lisans

Bu proje, **Bursa Teknik Üniversitesi Bilgisayar Mühendisliği** bölümü  
**BLM0463 - Veri Madenciliğine Giriş** dersi kapsamında,  
tamamen **eğitim ve akademik amaçlı** olarak geliştirilmiştir.  
Ticari bir kullanım amacı bulunmamaktadır.

---

## 👩‍💻 Geliştirici

**Sıdıka Fırat**  
🎓 Bursa Teknik Üniversitesi  
💻 Bilgisayar Mühendisliği  
📘 Ders: BLM0463 Veri Madenciliğine Giriş  
📅 Akademik Dönem: 2024–2025

---


