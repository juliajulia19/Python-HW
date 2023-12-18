from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

texts = []
path = '/Users/yuliakoltsova/Desktop/data_texts/'
for file in os.listdir(path):
    with open(path + file, encoding='utf-8', errors='ignore') as txt:
        text = txt.read()
        text = text.lower()
        text = re.sub('[^\w+ ]', '', text)
        text = re.sub('[0-9]+', '', text)
        text = [word for word in text.split() if not word in set(stopwords.words('russian'))]
        text = ' '.join(text)
        texts.append(text)

labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1]

dic = {'label' : labels, 'text' : texts}

df = pd.DataFrame(dic)
print(df)

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text']).toarray() #полученные вектора
cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray() #полученные вектора
y = df.iloc[:, 0].values #метки спама
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

prediction = dict()

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print(f"Точность модели MultinomialNB: {accuracy}")

classifier = GaussianNB()
classifier.fit(X_train, y_train)
prediction["GaussianNB"] = classifier.predict(X_test)
accuracy2 = accuracy_score(y_test,prediction["GaussianNB"])
#print(f"Точность модели GaussianNB: {accuracy2}")

model = RandomForestClassifier()
model.fit(X_train,y_train)
# Предсказание на решающих деревьях
prediction["RandomForrest"] = classifier.predict(X_test)
accuracy3 = accuracy_score(y_test,prediction["RandomForrest"])
print(f"Точность модели RandomForestClassifier(): {accuracy3}")

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
def tfidfs(data):
    number_of_texts = len(data)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    npm_tfidf = tfidf_matrix.todense()
    list_of_doc_vectors = []
    for index in range(number_of_texts):
        document_vector = npm_tfidf[index]
        list_of_doc_vectors.append(document_vector.tolist())

    tfidfs_for_texts = []
    types = tfidf_vectorizer.get_feature_names_out()
    for index in range(number_of_texts):
        types_tfidf = [(types[ind], list_of_doc_vectors[index][0][ind]) for ind in range(len(types))]
        tfidfs_for_texts.append(types_tfidf)

    return tfidfs_for_texts

sorted_tfidfs0 = sorted(tfidfs(texts)[0], key=lambda tup: tup[1], reverse=True)
print(sorted_tfidfs0[:10])

sorted_tfidfs1 = sorted(tfidfs(texts)[1], key=lambda tup: tup[1], reverse=True)
print(sorted_tfidfs1[:10])

sorted_tfidfs2 = sorted(tfidfs(texts)[2], key=lambda tup: tup[1], reverse=True)
print(sorted_tfidfs2[:10])

sorted_tfidfs3 = sorted(tfidfs(texts)[3], key=lambda tup: tup[1], reverse=True)
print(sorted_tfidfs3[:10])

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(texts)
top_nouns = sorted(tfidf_vectorizer.vocabulary_, key=lambda x: x[1], reverse=True)[:20]
print(top_nouns)