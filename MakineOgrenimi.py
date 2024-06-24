import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

stop_words = stopwords.words('turkish')


yazarlar = ['yazar1', 'yazar2' , 'yazar3' , 'yazar4' , 'yazar5']
veri = {'yazar': [], 'metin': []}


for yazar in yazarlar:
    for i in range(1, 21):
        dosya_adi = f"{yazar}_yazi_{i}.txt"
        dosya_yolu = os.path.join('C:\\Users\\KheanJaii\\Desktop\\veri_seti', dosya_adi)
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            metin = dosya.read()
            veri['yazar'].append(yazar)
            veri['metin'].append(metin)


df = pd.DataFrame(veri)


cv = CountVectorizer(max_features=1500, stop_words=stop_words)
X = cv.fit_transform(df['metin']).toarray()
y = df['yazar']


classifier = MultinomialNB()
classifier.fit(X, y)

#Test edeceğimiz metin buraya yazılacak.
yeni_metin = """


"""


yeni_X = cv.transform([yeni_metin]).toarray()

tahmin = classifier.predict(yeni_X)


#yazar1 Fatih ALTAYLI
#yazar2 Serdar Ali ÇELİKLER
#yazar3 Abbas GÜÇLÜ
#yazar4 Ahmet HAKAN
#yazar5 İlber ORTAYLI
print(f"Yazar Tahmini: {tahmin[0]}")