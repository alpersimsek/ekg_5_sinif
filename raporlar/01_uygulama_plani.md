# EKG 5 Etiketli Sınıflandırma Projesi — Uygulama Planı

**Tarih:** 2026-06-21  
**Veri kaynağı:** `/mnt/d/EKG_WORK/5_Class_Data`  
**Hedef:** Kilitli test kümesinde macro F1 ve accuracy değerlerinin 0.95 veya üzerine çıkarılması

## 1. Veri denetimi sonucu

| Özellik | Sonuç |
|---|---:|
| EKG kaydı | 573.355 |
| Hasta | 139.911 |
| Biçim | WFDB `.hea` + `.dat` |
| Sinyal | 12 derivasyon, 5.000 örnek |
| Süre / frekans | 10 saniye / 500 Hz |
| Tek etiketli kayıt | 533.233 |
| İki etiketli kayıt | 40.122 (%7,0) |

| Etiket | Kayıt | Prevalans |
|---|---:|---:|
| NORMAL | 461.098 | %80,4 |
| AFIB | 80.768 | %14,1 |
| AFL | 13.817 | %2,4 |
| LBBB | 23.567 | %4,1 |
| RBBB | 34.227 | %6,0 |

Etiketler birbirini dışlamamaktadır. Örneğin bir kayıt aynı anda `AFIB` ve `RBBB` olabilir. Bu nedenle birincil problem beş sınıflı softmax yerine beş çıkışlı **multi-label** sınıflandırma olarak tanımlanmalıdır.

## 2. Model çıktısı

Model her sınıf için bağımsız sigmoid olasılığı üretir:

```json
{
  "NORMAL": 0.02,
  "AFIB": 0.97,
  "AFL": 0.03,
  "LBBB": 0.08,
  "RBBB": 0.91
}
```

Sınıf eşikleri sadece validation kümesi üzerinde optimize edilir. Örnekte eşiklenmiş sonuç `AFIB + RBBB` olabilir.

## 3. Başarı kriterleri

Kilitli ve hasta bazında ayrılmış test kümesinde:

- Macro F1 >= 0.95
- Exact-match multi-label accuracy >= 0.95
- Tercihen her sınıf için F1 >= 0.90
- Hasta bazında bootstrap ile %95 güven aralıkları
- Train, validation ve test arasında sıfır hasta kesişimi

Ek metrikler: micro/weighted F1, Hamming accuracy/loss, sınıf bazında precision, recall, specificity, AUROC, average precision ve Brier score.

0.95 değeri bir kabul kapısıdır; baştan garanti edilemez. Otomatik kardiyoloji metinlerinden türetilen etiket gürültüsü ve özellikle AFIB/AFL ayrımı ulaşılabilecek skoru sınırlayabilir.

## 4. Uygulama aşamaları ve görev listesi

### Aşama 1 — Proje ve ortam

- Tekrarlanabilir Python ortamı ve kilitli bağımlılıklar
- Konfigürasyon tabanlı veri/model/çıktı yolları
- Seed, donanım, paket sürümü ve Git revizyon kaydı
- Unit/integration test altyapısı
- Deney takibi
- `/mnt/d` okuma hızının ölçülmesi ve yerel hızlı cache planı

### Aşama 2 — Veri ve etiket kalite denetimi

- Metadata'daki her WFDB kaydının okunabilirliğini doğrulama
- Derivasyon adı, sırası, frekans, uzunluk, gain ve birim kontrolü
- Eksik, düz, kırpılmış ve aşırı gürültülü sinyalleri belirleme
- Etiket birliktelik ve hasta frekans tabloları
- Her etiket kombinasyonundan tabakalı manuel örnek incelemesi
- Son kullanılabilir kayıt manifestinin sürümlenmesi

Teslimatlar: veri kalite raporu, etiket birliktelik ısı haritası, dışlama raporu ve temsilî EKG şekilleri.

### Aşama 3 — Leakage-safe veri bölme

`subject_id` bazında gruplanmış multi-label stratification uygulanır:

- Train: hastaların %70'i
- Validation: hastaların %15'i
- Kilitli test: hastaların %15'i

- Nadir AFL ve etiket birliktelik oranları korunur.
- Manifestler model geliştirmeden önce dondurulur.
- Hasta kesişimlerini engelleyen otomatik test eklenir.
- Preprocessing istatistikleri yalnızca train kümesinden hesaplanır.
- Geliştirme sırasında gerekirse hasta bazlı 3-fold cross-validation uygulanır.

### Aşama 4 — Sinyal pipeline'ı

1. WFDB ile fiziksel birimlerde okuma
2. Standart 12 derivasyon sırasına dönüştürme
3. Gerekirse baseline drift giderme ve band-pass filtreleme
4. Uç amplitüd değerlerini kontrollü kırpma
5. Train istatistikleriyle derivasyon bazında normalizasyon
6. Tam 10 saniyelik kaydı koruma
7. Hızlı, ardışık shard/cache biçimine dönüştürme

Yalnızca train için kontrollü augmentation: amplitüd ölçekleme, baseline wander, düşük Gaussian gürültü, zaman maskeleme, sınırlı zaman kaydırma ve derivasyon dropout. Ham EKG sinyallerine SMOTE uygulanmaz.

### Aşama 5 — Baseline modeller

- Etiket frekansı/majority baseline
- EKG özellikleri üzerinde lojistik model veya gradient boosting
- Küçük 1D CNN
- 1D ResNet
- InceptionTime tarzı multi-scale model

Tüm deneyler aynı split, preprocessing ve evaluation kodunu kullanır.

### Aşama 6 — Ana model

Başlangıç mimarisi:

- Girdi: `[batch, 12, 5000]`
- Multi-scale 1D convolution stem
- Residual/Inception blokları
- Squeeze-and-excitation veya derivasyon attention
- Global average pooling
- Beş sigmoid çıkış

Eğitim:

- İlk baseline için positive class weight içeren BCE loss
- Ardından focal veya asymmetric loss karşılaştırması
- Normal kayıtları atmadan dengeli batch sampling
- AdamW, warmup ve cosine learning-rate schedule
- Mixed precision ve gradient clipping
- Validation macro F1 ile early stopping
- Validation üzerinde sınıf bazlı threshold optimizasyonu
- Her deney için ham tahminlerin saklanması

Performans plato yaparsa: farklı receptive field'lar, 250/500 Hz karşılaştırması, rare-class sampling, self-supervised pretraining, uyumlu pretrained EKG encoder ve en iyi bağımsız modellerin ensemble edilmesi incelenir.

### Aşama 7 — Değerlendirme ve kanıt paketi

Multi-label model için tek 5x5 matris yerine her etiket için bir binary confusion matrix gerekir.

Üretilecek şekiller:

- Beş binary ve normalize confusion matrix
- Precision-recall ve ROC eğrileri
- Güven aralıklı sınıf bazlı F1 grafiği
- Threshold-F1 grafikleri
- Calibration/reliability eğrileri
- Etiket birliktelik ısı haritası
- Train/validation loss ve F1 eğrileri
- Etiket kombinasyonu ve sinyal kalitesine göre hata analizi
- Seçilmiş TP/FP/FN EKG örnekleri
- Derivasyon/zaman saliency görselleri

5x5 confusion matrix zorunluysa, yalnızca 533.233 tek etiketli kayıtla ayrı bir ikincil softmax deneyi yapılır. Bu sonuç birincil multi-label modelin sonucu olarak sunulmaz.

### Aşama 8 — Hata analizi ve iyileştirme

- AFIB/AFL hatalarını ayrı inceleme
- NORMAL ile birlikte bulunan LBBB/RBBB örneklerini inceleme
- Model hatası ile muhtemel etiket hatasını ayırma
- Confidence dağılımları ve sinyal kalite etkisi
- Loss, augmentation, sampling ve mimari ablation deneyleri
- Seçilen modelleri birden fazla seed ile yeniden eğitme

Her değişikliğin macro F1, exact-match accuracy ve nadir sınıf recall etkisi deney tablosunda tutulur.

### Aşama 9 — Üretim paketi

- Dondurulmuş preprocessing pipeline'ı ve model checkpoint'i
- WFDB kaydı kabul eden CLI/API
- Beş olasılık, thresholded etiketler ve kalite uyarıları
- ONNX veya TorchScript export
- CPU/GPU gecikme ve bellek benchmark'ı
- Kilitli ortam veya container tanımı
- Batch prediction ve otomatik rapor üretim komutları
- Model card, sınırlamalar ve intended-use dokümanı

### Aşama 10 — Final doğrulama

- Kilitli test final model için bir kez değerlendirilir.
- Hasta bazlı bootstrap güven aralıkları raporlanır.
- Tüm grafikler kaydedilmiş tahminlerden tekrar üretilebilir olur.
- Leakage ve temiz ortam inference testleri geçer.
- Harici veriyle test yapılmadıkça sonuç "internal MIMIC validation" olarak tanımlanır.
- Klinik kullanım için bağımsız harici doğrulama ve uzman incelemesi gerekir.

## 5. Önerilen uygulama sırası

1. Ortam ve repository iskeleti
2. Veri kalite raporu
3. Dondurulmuş hasta bazlı split'ler
4. Cache/shard preprocessing pipeline'ı
5. Küçük CNN ve ResNet baseline
6. Inception/residual ana model
7. Threshold calibration ve hata analizi
8. Gerekirse ensemble
9. Kilitli test değerlendirmesi
10. Grafikler, inference paketi ve final rapor

## 6. Karar verilmesi gereken konular

1. Birincil sistemin multi-label olarak onaylanması
2. Accuracy tanımının exact-match accuracy olarak kabul edilmesi
3. 5x5 softmax deneyinin yalnızca ikincil rapor olarak istenip istenmediği
4. Kullanılabilir GPU modeli ve VRAM miktarı
5. Cache için hızlı yerel disk kapasitesi
6. Deney takibi için MLflow veya Weights & Biases tercihi
7. Harici doğrulama veri seti ve klinik uzman erişimi

## 7. Mevcut donanım ve depolama

2026-06-21 tarihinde yapılan ortam kontrolü:

| Kaynak | Sonuç |
|---|---:|
| GPU | NVIDIA GeForce RTX 3090 |
| GPU belleği | 24 GB (24.576 MiB) |
| CUDA compute capability | 8.6 |
| CPU | 28 logical core |
| Sistem belleği | 23 GiB |
| Swap | 6 GiB |
| WSL ext4 boş alan | 797 GB |
| `/mnt/d` boş alan | 715 GB |
| Ham `.dat` sinyallerinin tahminî boyutu | 64,1 GiB |

WSL GPU aygıtı `/dev/dxg` üzerinden görünmektedir. Linux tarafında `nvidia-smi` komutu kurulu değildir; Windows `nvidia-smi.exe` GPU ve sürücü bilgilerini başarıyla raporlamıştır. PyTorch kurulumundan sonra CUDA ile gerçek tensor testi yapılmalıdır.

Preprocessing ve deneyler için WSL ext4 üzerinde 180–200 GB alan ayrılması önerilir. Varsayılan cache konumu `/home/alper/EKG_CACHE` olacaktır. Orijinal veri `/mnt/d` altında salt okunur kaynak gibi kullanılmalıdır. Float32 cache yaklaşık 128 GiB, float16 veya int16 cache yaklaşık 64 GiB gerektirir.

RTX 3090 ana 1D ResNet/Inception modelini eğitmek için yeterlidir. İlk GPU deneyi mixed precision ve batch size 64 ile başlatılacak; ölçülen VRAM kullanımına göre batch size ayarlanacaktır. 23 GiB sistem belleği nedeniyle tüm veri RAM'e alınmayacak, shard tabanlı streaming kullanılacaktır.

## 8. Python kütüphane yığını

- Model: PyTorch CUDA 12.8 build
- Sinyal/veri: WFDB, NumPy, pandas, SciPy, scikit-learn
- Multi-label split: iterative-stratification
- Konfigürasyon: Hydra ve Pydantic
- Deney takibi: yerel MLflow
- Grafik/açıklanabilirlik: Matplotlib, Seaborn, Plotly ve Captum
- Cache: HDF5 ve Zarr/Numcodecs
- Kalite: pytest, Ruff ve mypy
- Export/servis: ONNX, ONNX Runtime, FastAPI ve Uvicorn

Kurulum girdileri proje kökündeki `requirements.txt` dosyasında tutulur. Kurulum doğrulamasında paket import'larına ek olarak CUDA tensor işlemi ve WFDB örnek kayıt okuması test edilmelidir.
