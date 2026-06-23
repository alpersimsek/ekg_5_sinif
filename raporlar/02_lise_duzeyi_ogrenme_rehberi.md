# EKG Yapay Zekâ Projesi — Lise Düzeyi Öğrenme Rehberi

**Tarih:** 22 Haziran 2026  
**Proje:** Beş EKG bulgusunu tanıyan çok etiketli yapay zekâ sistemi  
**Durum:** İlk tam eğitim tamamlandı; doğrulama sonuçları üzerinde hata analizi ve
iyileştirme deneyleri planlanıyor. Kilitli test kümesi henüz kullanılmadı.

## 1. Bu projede ne yapıyoruz?

Kalp her attığında elektriksel bir etkinlik oluşur. EKG (elektrokardiyogram), bu
elektriksel etkinliği vücudun farklı noktalarından ölçerek çizgi biçiminde gösterir.
Doktorlar bu çizgilerin şekline, hızına ve düzenine bakarak kalple ilgili bazı
durumları anlayabilir.

Bu projede bir bilgisayara EKG kayıtlarını incelemeyi öğretiyoruz. Sistem, bir
kayıtta şu beş bulgunun bulunma olasılığını hesaplayacak:

- `NORMAL`: İncelenen beş bulgu açısından normal kayıt
- `AFIB`: Atriyal fibrilasyon
- `AFL`: Atriyal flutter
- `LBBB`: Sol dal bloğu
- `RBBB`: Sağ dal bloğu

Bir EKG'de aynı anda birden fazla bulgu bulunabilir. Örneğin hem AFIB hem RBBB
görülebilir. Bu nedenle sistem yalnızca bir sınıf seçmez; beş bulgunun her biri
için ayrı bir olasılık üretir. Buna **çok etiketli sınıflandırma** denir.

## 2. Neden yapıyoruz?

Yüz binlerce EKG kaydını tek tek incelemek çok zaman alır. İyi eğitilmiş bir model:

- çok sayıda kaydı hızlı biçimde ön değerlendirmeden geçirebilir,
- şüpheli kayıtların önceliklendirilmesine yardımcı olabilir,
- aynı ölçütleri her kayda tutarlı biçimde uygulayabilir,
- araştırmacıların büyük veri kümelerinde örüntü bulmasını kolaylaştırabilir.

Bu sistem doktorun yerini almak için değil, karar desteği ve araştırma amacıyla
geliştirilmektedir. Klinik kullanım için farklı hastanelerden veriyle doğrulama,
uzman değerlendirmesi ve ilgili yasal süreçler gerekir.

## 3. Kullandığımız veri nedir?

Ham MIMIC-IV-ECG dizininde **800.035 EKG kaydı** vardır. Bunların 573.355'i
seçilen beş etiketten en az biriyle eşleşmiştir. Geri kalan **226.680 kayıt** ise
sinüs bradikardisi, sinüs taşikardisi, uzamış QT, pacemaker ritmi, eksen sapmaları,
hipertrofi, infarkt ve ST-T değişiklikleri gibi başka bulgular içermektedir.

Beş sınıflık çalışma için çıkarılan veri kümesinde **573.355 EKG kaydı** ve
**139.911 hasta** vardır. Her kayıt:

- 10 saniye uzunluğundadır,
- saniyede 500 ölçüm içerir,
- toplam 5.000 zaman noktasından oluşur,
- kalbi 12 farklı açıdan gösteren 12 derivasyona sahiptir.

Bir kayıt, 12 satır ve 5.000 sütundan oluşan büyük bir sayı tablosu gibi
düşünülebilir. Böylece tek kayıtta 60.000 ölçüm bulunur.

Etiketlerin dağılımı eşit değildir. NORMAL kayıtlar çok fazla, AFL kayıtları ise
azdır. Bu duruma **sınıf dengesizliği** denir. Model yalnızca en sık görülen sınıfı
öğrenmesin diye eğitim sırasında az görülen sınıflara daha fazla önem veren bir
kayıp hesabı kullanacağız.

## 4. Çalışmanın ana aşamaları

### Aşama A — Veriyi anlamak ve denetlemek

Önce her EKG dosyasının gerçekten okunup okunamadığını kontrol ediyoruz. Ayrıca:

- 12 derivasyonun eksiksiz olup olmadığına,
- kayıt uzunluğuna ve örnekleme hızına,
- eksik veya sayı olmayan ölçümlere,
- tamamen düz kalan derivasyonlara,
- olağan dışı büyük genliklere bakıyoruz.

Bu adım önemlidir; bozuk veriler modele verilirse model kalp örüntüleri yerine veri
hatalarını öğrenebilir. İlk rastgele 1.000 kaydın denetiminde 982 kayıt geçerli,
18 kayıt geçersiz bulundu. Daha sonra 573.355 kaydın tamamı denetlendi: 565.550
kayıt geçerli, 7.805 kayıt geçersiz bulundu. Geçersiz oranı yaklaşık `%1,36` oldu.
Başlıca nedenler sonlu olmayan değerler, düz derivasyonlar ve aşırı genliklerdi.

### Aşama B — Veriyi eğitim, doğrulama ve test olarak ayırmak

Veriyi üç parçaya ayırıyoruz:

- `%70` eğitim: modelin öğrendiği bölüm
- `%15` doğrulama: ayarların seçildiği bölüm
- `%15` test: en sonda başarıyı tarafsız ölçtüğümüz kilitli bölüm

Aynı hastanın farklı EKG'leri farklı bölümlere konulmaz. Aksi hâlde model, testte
daha önce gördüğü hastayı tanıyabilir ve gerçekte olduğundan yüksek başarı verir.
Bu hataya **veri sızıntısı (data leakage)** denir.

Test kümesini model geliştirme sırasında kullanmıyoruz. Sınava çalışırken cevap
anahtarına bakmak nasıl gerçek başarıyı bozarsa, test sonuçlarına göre modeli
ayarlamak da bilimsel ölçümü bozar.

### Aşama C — Sinyali hazırlamak

Ham EKG sinyallerine şu işlemleri uyguluyoruz:

1. Dosyayı fiziksel milivolt biriminde okuma
2. Derivasyonları standart sıraya koyma
3. Çok yavaş taban kaymasını ve yüksek frekanslı gürültüyü filtreleme
4. Aşırı uç değerleri güvenli aralıkta kırpma
5. Her derivasyonu eğitim verisinin ortalama ve standart sapmasıyla ölçekleme

Ortalama ve standart sapma yalnızca eğitim bölümünden hesaplanır. Doğrulama veya
test verisinden bilgi kullanmak yine veri sızıntısı oluştururdu.

### Aşama D — Modeli eğitmek

Kullandığımız model bir **1 boyutlu evrişimli sinir ağıdır (1D CNN)**. Evrişim
katmanları EKG üzerindeki kısa ve uzun şekilleri arar. Residual bağlantılar derin
bir ağın daha kararlı öğrenmesine yardım eder. Squeeze-and-excitation mekanizması
ise modelin hangi özellik kanallarına daha fazla önem vereceğini öğrenir.

Modelin beş çıkışı vardır. Her çıkış `0` ile `1` arasında bir olasılığa çevrilir.
Örneğin:

```text
NORMAL: 0,02
AFIB:   0,97
AFL:    0,03
LBBB:   0,08
RBBB:   0,91
```

Bu örnekte model AFIB ve RBBB'nin birlikte bulunduğunu düşünüyor. Bir olasılığın
etikete dönüşmesi için kullanılan sınır değerine **eşik** denir. Her sınıfın eşiği
yalnızca doğrulama verisi üzerinde seçilir.

### Aşama E — Başarıyı ölçmek

Tek bir başarı sayısı yeterli değildir. Şu ölçümleri birlikte kullanıyoruz:

- **Precision:** Modelin “var” dediklerinin ne kadarı gerçekten var?
- **Recall:** Gerçek vakaların ne kadarını model buldu?
- **F1:** Precision ve recall değerlerini dengeleyen ölçü
- **Macro F1:** Beş sınıfın F1 değerlerinin eşit ağırlıklı ortalaması
- **Exact-match accuracy:** Bir kayıttaki beş etiketin tamamı doğru mu?
- **AUROC ve average precision:** Farklı eşiklerde ayırt etme başarısı
- **Brier score:** Üretilen olasılıkların ne kadar güvenilir olduğu

Özellikle AFL az görüldüğü için yalnızca genel accuracy değerine bakmak yanıltıcı
olabilir. Model NORMAL kayıtları iyi bilip AFL kayıtlarını kaçırsa bile yüksek bir
genel sayı elde edebilir. Macro F1 her sınıfa eşit önem vererek bu sorunu azaltır.

Hedefimiz kilitli test kümesinde Macro F1 ve exact-match accuracy değerlerinin
`0,95` veya üzerine çıkmasıdır. Bu bir hedef ve kabul ölçütüdür; veri ve etiket
kalitesi görülmeden garanti edilemez.

## 5. Şu ana kadar ne yaptık?

- Proje yapısı ve tekrar üretilebilir ayarlar oluşturuldu.
- Python, PyTorch ve CUDA ortamı doğrulandı.
- NVIDIA RTX 3090 ekran kartı PyTorch tarafından başarıyla görüldü.
- 573.355 kayıt hasta kimliğine göre eğitim, doğrulama ve test kümelerine ayrıldı.
- İlk ayrımda hasta sızıntısı olmadığı otomatik olarak doğrulandı.
- Sinyal okuma, filtreleme, veri kümesi, model, metrik ve eşik kodları yazıldı.
- 14 otomatik test geçti; kod kalite denetimi hata vermedi.
- İlk kalite taraması bozuk sinyallerin manifestlerden çıkarılması gerektiğini gösterdi.
- Tam veri kalite taraması sekiz paralel işlemle 26 dakika 10 saniyede tamamlandı.
- 7.805 geçersiz kayıt çıkarıldı ve manifestler 565.550 geçerli kayıtla yenilendi.
- 256 kayıtlık izole normalizasyon ve GPU eğitim denemesi uçtan uca tamamlandı.
- 396.128 temiz eğitim kaydının gerçek normalizasyon değerleri hesaplandı.
- 4.096 kayıtlık GPU benchmark'ında 21 epoch 203 saniyede tamamlandı.
- 396.128 eğitim ve 84.063 doğrulama kaydıyla ilk tam model eğitildi.
- Eğitim, doğrulama Macro F1 değeri sekiz epoch boyunca iyileşmeyince 22. epoch'ta
  otomatik olarak durdu. En iyi model 14. epoch'ta elde edildi.
- En iyi doğrulama Macro F1 değeri `0,86217`, exact-match accuracy değeri ise
  `0,93387` oldu.
- İki yapılandırılmış çıkış başlığı kodlandı ve 256 kayıtlık uçtan uca duman
  testini başarıyla tamamladı.
- İki başlıklı tam eğitim 19. epoch'ta erken durdurmayla tamamlandı. En iyi
  doğrulama sonucu 11. epoch'ta elde edildi: Macro F1 `0,86308`, exact-match
  accuracy `0,93865` ve AFL F1 `0,66225`.
- En iyi iki başlıklı model `0,0001` öğrenme hızıyla yeniden başlatıldı. Doğrulama
  sonucu durduğunda öğrenme hızını azaltan ikinci iyileştirme deneyi devam ediyor.

## 6. Neden paralel işlem kullanıyoruz?

573.355 dosyayı tek bir işlemle sırayla okumak saatler sürebilir. Paralel işlemde
bilgisayar aynı anda birkaç kaydı inceler. Sekiz işçi, sekiz ayrı görevli gibi
düşünülebilir. İlk uygulama saniyede yaklaşık 60–70 kayıt denetlerken paralel sürüm
çoğu anda saniyede yaklaşık 350–450 kayıt denetleyebilmektedir. Disk hızı ve bazı
dosyaların daha yavaş okunması nedeniyle hız sabit değildir.

## 7. Sonraki adımlar

1. İlk modelin yanlış pozitif ve yanlış negatif örneklerini sınıf bazında incelemek
2. Etiket yapısını kullanan iki çıkışlı modeli eğitmek
3. AFL için kayıp hesabı ve örnek seçme yöntemlerini karşılaştırmak
4. Öğrenme hızı ve düzenlileştirme deneyleri yapmak
5. Ritim ve EKG şekil bilgisini daha iyi kullanan model değişikliklerini denemek
6. En iyi ayarları en az üç farklı rastgele başlangıçla doğrulamak
7. En güçlü modelleri birleştiren bir ensemble denemek
8. Bütün kararlar bittikten sonra kilitli test kümesini yalnızca bir kez değerlendirmek
9. Hasta düzeyinde güven aralıkları, grafikler ve hata analizi üretmek

## 8. Sonuçlar bölümü

Bu bölüm işlemler tamamlandıkça güncellenecektir.

| Kontrol | Güncel sonuç |
|---|---:|
| Ham MIMIC-IV-ECG kaydı | 800.035 |
| Beş hedef sınıfa eşleşen | 573.355 |
| Beş hedef sınıf dışında kalan | 226.680 |
| Toplam kayıt | 573.355 |
| Toplam hasta | 139.911 |
| Örnek kalite denetimi | 1.000 kayıt |
| Örnekte geçerli kayıt | 982 |
| Örnekte geçersiz kayıt | 18 |
| Otomatik test | 14/14 başarılı |
| Kod kalite kontrolü | Başarılı |
| GPU | NVIDIA GeForce RTX 3090 |
| Tam kalite denetimi | 573.355/573.355 tamamlandı |
| Geçerli kayıt | 565.550 |
| Dışlanan kayıt | 7.805 (%1,36) |
| Temiz eğitim / doğrulama / test | 396.128 / 84.063 / 85.359 |
| Küçük GPU duman testi | 16 epoch, başarıyla tamamlandı |
| Gerçek normalizasyon | 396.128/396.128 tamamlandı |
| 4.096 kayıt benchmark Macro F1 | 0,80 (nihai sonuç değildir) |
| İlk tam eğitim | 22 epoch; erken durdurma ile tamamlandı |
| En iyi doğrulama epoch'u | 14 |
| Doğrulama Macro F1 | 0,86217 |
| Doğrulama exact-match accuracy | 0,93387 |
| İki başlıklı model doğrulama Macro F1 | 0,86308 |
| İki başlıklı model exact-match accuracy | 0,93865 |
| Düşük öğrenme hızlı ince ayar | Tam veriyle eğitim devam ediyor |
| Kilitli test sonucu | Henüz yok |

### İlk tam modelin sınıf bazındaki doğrulama sonuçları

| Sınıf | Gerçek pozitif kayıt | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| NORMAL | 68.113 | 0,979 | 0,986 | 0,983 |
| AFIB | 11.404 | 0,896 | 0,906 | 0,901 |
| AFL | 2.017 | 0,642 | 0,667 | 0,654 |
| LBBB | 3.350 | 0,867 | 0,917 | 0,892 |
| RBBB | 4.797 | 0,873 | 0,889 | 0,881 |

Bu tablo Macro F1 değerini en çok AFL sınıfının sınırladığını gösteriyor. AFL,
doğrulama kayıtlarının yalnızca yaklaşık `%2,4`'ünde bulunuyor. Model 672 AFL
kaydını kaçırdı ve AFL olmayan 750 kayda yanlışlıkla AFL dedi. Bu 750 yanlış
pozitif kaydın 565'i gerçekte AFIB idi. Yani temel sorunlardan biri yalnızca az
örnek olması değil, AFIB ile AFL ritimlerinin birbirine karıştırılmasıdır.

## 9. Macro F1 değerini nasıl yükseltebiliriz?

### 9.1. Önce etiketlerin yapısını modele öğretmek

Şimdiki model beş etiketi birbirinden tamamen bağımsız beş evet/hayır sorusu gibi
öğreniyor. Oysa temiz veri kümesinde NORMAL, AFIB ve AFL aynı ritim grubundadır;
bir kayıtta bu üçünden en fazla biri bulunur. Benzer biçimde LBBB ve RBBB aynı
kayıtta birlikte bulunmamaktadır.

Bir sonraki modelde ortak EKG kodlayıcısının ardından iki ayrı karar başlığı
kullanılacaktır:

1. Ritim başlığı: `yok`, `NORMAL`, `AFIB` veya `AFL`
2. İleti başlığı: `yok`, `LBBB` veya `RBBB`

İki başlık birlikte çalıştığı için bir kayıt örneğin hem AFIB hem RBBB olabilir;
ancak aynı anda AFIB ve AFL denmesi engellenir. Bu değişiklik özellikle 565
AFIB→AFL karışıklığını azaltmayı hedefler. İlk ve en önemli yeniden eğitim deneyi
budur.

### 9.2. AFL sınıfına odaklanan kayıp ve örnek seçimi

Mevcut eğitimde az bulunan her sınıfa otomatik olarak büyük ağırlık veriliyor.
Bu yöntem AFL'yi görünür kılıyor fakat yanlış pozitifleri de artırabiliyor. Şu
yöntemler ayrı deneylerde karşılaştırılacaktır:

- AFL içeren hastaların eğitim mini-batch'lerinde kontrollü olarak daha sık görülmesi,
- kolay negatif örneklerin etkisini azaltan focal veya asymmetric loss,
- sınıf sıklığını daha yumuşak kullanan logit adjustment.

Bu yöntemler aynı anda değiştirilmemelidir. Her deney tek bir değişiklik içermeli,
aynı hasta ayrımı ve aynı doğrulama ölçütüyle karşılaştırılmalıdır.

### 9.3. Öğrenme hızını daha iyi yönetmek

En iyi sonuç 14. epoch'ta geldi. Eğitim kaybı daha sonra azalmaya devam ederken
doğrulama kaybı yükseldi. Bu, modelin eğitim örneklerine gereğinden fazla uyum
sağlamaya başladığını gösterir. En iyi modelden daha düşük öğrenme hızıyla kısa
bir ince ayar ve doğrulama sonucu durduğunda öğrenme hızını azaltan bir zamanlayıcı
denenecektir. Dropout ve weight decay de küçük, kontrollü aralıklarla taranacaktır.

### 9.4. Ritim ve şekil bilgisini birlikte kullanmak

AFIB ve AFL ayrımında yalnızca tek bir kalp atımının şekli değil, atımların zaman
aralıkları da önemlidir. İleri deneylerde R tepeleri eğitim verisinden otomatik
bulunarak atımlar arası süreler yardımcı özellik olarak verilebilir. Ayrıca kısa
ve uzun EKG örüntülerini aynı anda gören çok ölçekli evrişimler denenebilir.
R-tepesi bulma hataları yeni bir hata kaynağı olabileceği için bu adım, iki başlıklı
modelden sonra uygulanacaktır.

### 9.5. Eşik, tekrar ve ensemble

Mevcut doğrulama tahminlerinde daha ayrıntılı eşik araması Macro F1 değerini
yalnızca yaklaşık `0,86217`'den `0,86247`'ye çıkarıyor. Bu nedenle yalnızca eşik
değiştirmek yeterli değildir. Yine de son modelde eşikler ayrı bir kalibrasyon
bölümünde seçilecektir.

Bir deneyin iyi görünmesi rastgele başlangıçtan kaynaklanabilir. En iyi ayar en az
üç farklı seed ile eğitilecek; ortalama ve değişkenlik raporlanacaktır. Son olarak
bu modellerin olasılık ortalaması alınarak ensemble oluşturulabilir. Ensemble
genellikle küçük fakat daha güvenilir bir artış sağlar; karşılığında çalışma
maliyeti yükselir.

### 9.6. Deney sırası ve başarı ölçütü

| Sıra | Deney | Ana soru |
|---:|---|---|
| 0 | Mevcut model | Karşılaştırma tabanı nedir? |
| 1 | İki yapılandırılmış çıkış başlığı | AFIB–AFL karışıklığı azalıyor mu? |
| 2 | Daha düşük öğrenme hızı / yeni zamanlayıcı | Aşırı uyum gecikiyor mu? |
| 3 | AFL odaklı loss veya örnekleme | AFL F1 artarken diğer sınıflar korunuyor mu? |
| 4 | Ritim özellikleri / çok ölçekli model | Zor ritim örnekleri daha iyi ayrılıyor mu? |
| 5 | Üç seed ve ensemble | Kazanç tekrarlanabilir ve kararlı mı? |

Her deneyde ana ölçüt doğrulama Macro F1 olacaktır. Ayrıca sınıf bazında F1,
özellikle AFL precision ve recall, birlikte izlenecektir. Kilitli test kümesi bu
deneylerin hiçbirinde ayar seçmek için kullanılmayacaktır.

## 10. Önemli bilimsel ve etik sınırlar

- Etiketler otomatik raporlardan türetildiyse yanlış veya eksik olabilir.
- AFIB ve AFL birbirine benzeyen ritimler olduğu için ayrımları zor olabilir.
- Tek bir veri kümesindeki yüksek başarı başka hastanelerde aynı sonucu garanti etmez.
- Modelin yaş, cinsiyet veya cihaz türü gibi gruplarda farklı davranıp davranmadığı
  ayrıca incelenmelidir.
- Sonuçlar bir kardiyoloji uzmanının klinik kararının yerine geçmez.

## 11. Kısa sözlük

| Kavram | Basit açıklama |
|---|---|
| EKG | Kalbin elektriksel etkinliğinin kaydı |
| Derivasyon | Kalbe farklı bir açıdan bakan ölçüm kanalı |
| Etiket | Kayıtta bulunduğu söylenen durum |
| Model | Veriden örüntü öğrenen matematiksel sistem |
| Eğitim | Modelin örneklerden öğrenme süreci |
| Doğrulama | Ayar ve eşik seçmek için kullanılan ayrı veri |
| Test | En sonda tarafsız başarı ölçümü yapılan kilitli veri |
| Epoch | Modelin eğitim verisini bir kez işlemesi |
| Kayıp (loss) | Model hatasını sayıya dönüştüren ölçü |
| Eşik | Olasılığı “var/yok” kararına dönüştüren sınır |
| Veri sızıntısı | Test bilgisinin yanlışlıkla öğrenme sürecine karışması |
| Overfitting | Modelin genel kural yerine eğitim örneklerini ezberlemesi |
