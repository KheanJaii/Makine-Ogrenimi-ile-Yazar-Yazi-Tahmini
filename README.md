# Makine Ogrenimi ile Yazar Yazi Tahmini
 Yazarlara ait köşe yazıları ile eğitilmesi ve yeni yazıların yazarlarının tahmin edilmesini sağlayan model

 Projenin amacı: Yazarlara ait belli miktardaki köşe yazısı ile model
eğitilerek veri setinde bulunmayan köşe yazısı modele sorulduğunda doğru
bir şekilde yazarının tahmin edilmesi.

Projenin gerçeklenmesi:

os: Dosya işlemleri yapmak için kullanılır. Dosya adı oluşturma ve dosya
yollarını birleştirme gibi işlemlerde kullanılır.
● pandas: Veri manipülasyonu ve analizi için kullanılır. Veriyi daha etkili bir
şekilde işlemek ve düzenlemek için kullanılır.
● CountVectorizer: Metin madenciliği ve doğal dil işleme uygulamalarında
kullanılan bir araçtır. Metin verisini sayısal bir özellik vektörüne dönüştürmek
için kullanılır.
● MultinomialNB: Naive Bayes sınıflandırma modelini uygulamak için kullanılır.
Ön işleme adımı:
● stopwords: Doğal dil işleme uygulamalarında yaygın olarak kullanılan
stop-words listelerine erişim sağlar.
● stopwords.words('turkish'): Türkçe stop-words listesini sağlayan
nltk kütüphanesinin stopwords modülünü kullanır.
● stop_words: Türkçe stop-words listesini içeren bir liste oluşturur. Bu liste,
genelde analiz sırasında göz ardı edilmesi gereken yaygın Türkçe kelimeleri
içerir. Örneğin, "ve", "veya", "ama", "bir", "şu", gibi kelimeler bu liste içinde
olabilir.
Stop-words listesini kullanmanın amacı, modelin eğitim ve tahmin aşamalarında,
anlam taşımayan ve genellikle sık kullanılan kelimelerin modele fazla etki etmesini
önlemektir. Bu kelimeler modelin genelleme yeteneğini artırabilir ve modelin daha
spesifik ve anlamlı özelliklere odaklanmasını sağlar.

CountVectorizer sınıfında stop_words parametresine bu stop-words listesini
vererek, metin içinde geçen ancak model için önemsiz olan kelimelerin çıkarılmasını
sağlamış oluruz.

pd.DataFrame(veri) ifadesi, Python'da popüler bir veri analizi kütüphanesi olan
Pandas'ı kullanarak, önceden oluşturulan veri sözlüğünü bir DataFrame'e
dönüştürmek içindir. DataFrame, tablo şeklinde veri yapısını temsil eder ve veri
analizi, manipülasyonu ve model eğitimi için kullanılmak üzere çok kullanışlıdır.
Önceki kod parçasında veri adlı bir sözlük oluşturuldu. Bu sözlükte 'yazar' ve 'metin'
adlı iki anahtar bulunuyordu. Her bir yazar için metin verilerini içeren bu sözlük, daha
sonra pd.DataFrame() fonksiyonuyla bir DataFrame'e dönüştürülür.
Bu işlem, veriyi düzenlemek ve makine öğrenimi modelinin kullanabileceği bir
formata getirmek için önemlidir. DataFrame, veriyi düzenlemek, sorgulamak,
görselleştirmek ve modele uygun bir formata getirmek için kullanılan bir araçtır.
Ayrıca, bir DataFrame üzerinde çeşitli Pandas fonksiyonları kullanılarak veri analizi
yapılabilir.


CountVectorizer, metin verisini sayısal özellik vektörlerine dönüştürmek için
kullanılır.
● max_features=1500: En fazla 1500 özellik kullanılacak şekilde sınırlar. Yani, en
sık geçen 1500 kelimeyi kullanarak her bir metni temsil eden bir özellik
vektörü oluşturulacaktır.
● stop_words=stop_words: Daha önce tanımlanan Türkçe stop-words listesini
kullanarak, modelin eğitiminde göz ardı edilecek kelimeleri belirtir.
● cv.fit_transform(df['metin']): CountVectorizer'ı kullanarak metin
verilerini özellik vektörlerine dönüştürür. Her bir metin, kelime frekanslarını
içeren bir özellik vektörüne dönüştürülür.
● toarray(): Elde edilen seyrek matrisi (sparse matrix) bir NumPy array'ine
dönüştürür. Çoğu makine öğrenimi modeli, yoğun matrix formunu kullanır, bu
nedenle bu dönüşüm yapılır.
● Sonuç olarak, X değişkeni, her bir satırı bir köşe yazısını temsil eden ve
sütunları kelime frekanslarını içeren bir matrisi temsil eder.
● y = df['yazar']: Yazar adlarını içeren 'yazar' sütununu y değişkenine atar.
Bu, her bir köşe yazısının hangi yazar tarafından yazıldığını temsil eden
etiketleri içerir.

classifier = MultinomialNB(): Multinomial Naive Bayes sınıflandırıcı
modelini oluşturur. Naive Bayes sınıflandırıcılar, özellikler arasındaki bağımsızlık
varsayımına dayalı olarak çalışan olasılık temelli modellerdir. "Multinomial" ifadesi,
bu modelin çok sınıflı (yazar sayısı kadar) sınıflandırma problemleri için uygun
olduğunu belirtir.
classifier.fit(X, y): Modeli eğitir.
X: Özellik matrisini temsil eder. Bu matris, metin verilerini sayısal özellik
vektörlerine dönüştürmekte kullanılan matristir (X değişkeni).
y: Etiket vektörünü temsil eder. Bu vektör, her bir köşe yazısının hangi
yazar tarafından yazıldığını belirtir (y değişkeni).

Model, bu özellik matrisi ve etiket vektörü üzerinde eğitilir. Eğitim süreci, modelin
metin verileriyle ilgili desenleri öğrenmesini sağlar. Naive Bayes modeli, sınıflar
arasındaki olasılıkları ve özellikler arasındaki ilişkileri öğrenir.
Bu eğitim sürecinden sonra, model artık yeni ve bilinmeyen metinlerin hangi yazar
tarafından yazıldığını tahmin edebilir. Bu tahminler, modelin öğrendiği olasılıkları ve
özellikler arasındaki ilişkileri kullanarak yapılır. Bu nedenle, eğitim süreci önemlidir
çünkü model, belirli bir yazarın yazma tarzını temsil eden özellikleri öğrenir.

Bu kod parçası, eğitilmiş CountVectorizer nesnesini kullanarak yeni bir köşe yazısını
sayısal özellik vektörüne dönüştürmek için kullanılır.

Sonuç: Beş farklı yazara ait eğitim veri setinde bulunmayan toplam on
beş köşe yazısı modele sorulduğunda tamamını doğru yazarla
eşleştirmeyi başarmıştır. Eğitim setinde mevcut olmayan köşe yazıları ek
dosya şeklinde projeye eklenmiştir. İstenilen köşe yazısı direkt olarak kod
içerisinde 36. satırda yeni_metin adlı değişkene atanarak test edilebilir.
