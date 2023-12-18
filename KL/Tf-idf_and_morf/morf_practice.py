import re
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from nltk.tokenize import word_tokenize
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
import collections
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
import spacy
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier


with open('/Users/yuliakoltsova/Desktop/data_texts/hud2', encoding = 'utf-8') as f:
    text = f.read()
    text = re.sub('[^\w ]+', '', text)
    text = text.lower()
    filtered_list = []
    tokenized_text = word_tokenize(text)
    for word in tokenized_text:
        if word not in stop_words:
            filtered_list.append(word)
    text = ' '.join(filtered_list)

text_tokens_nltk = word_tokenize(text)
#print(text_tokens_nltk)

mystem = Mystem()
text_analyzed_ms = mystem.analyze(text)  #много морфологической информации
#print(text_analyzed_ms)

#print(text_analyzed_ms[4])
#print(type(text_analyzed_ms[4]))

text_lemmatized_ms = mystem.lemmatize(text)
#print(text_lemmatized_ms)

#print('Слово - ', text_analyzed_ms[0]['text'])
#print('Разбор слова - ', text_analyzed_ms[0]['analysis'][0])
#print('Лемма слова - ', text_analyzed_ms[0]['analysis'][0]['lex'])
#print('Грамматическая информация слова2 - ', text_analyzed_ms[0]['analysis'][0]['gr'])

tagged_nltk1 = nltk.pos_tag(text_tokens_nltk) #очень плохо разметил - почти все токены 'NNP'
#print(tagged_nltk1)

tagged_nltk2 = nltk.pos_tag(text_tokens_nltk, lang='rus') #достойная частеречная разметка, tagset='universal'для английского
#print(tagged_nltk2)


list_of_tagged_nltk = []
for elem in tagged_nltk2:
    tag_tog = '_'.join(elem)
    list_of_tagged_nltk.append(tag_tog)
#print(list_of_tagged_nltk)

#new_text_tagged = ' '.join(['_'.join(elem) for elem in tagged_nltk2])


#найти слова по количеству частей речи
#number_of_pos = re.findall('[а-яА-Я]+_A=f', new_text_tagged)
#print(f'Your text has {len(number_of_pos)} adjectives.')

frequency_distribution_1 = FreqDist(text_tokens_nltk)
# print(frequency_distribution)
#print(frequency_distribution_1.most_common(100))
#frequency_distribution_1.plot(30, cumulative=False)

frequency_distribution_2 = FreqDist(list_of_tagged_nltk)
#print(frequency_distribution_2.most_common(100))
#frequency_distribution_2.plot(30, cumulative=False)

#в целом очень похожие списки, тк достаточно простой текст

list_1 = frequency_distribution_1.most_common(100)
list_2 = frequency_distribution_2.most_common(100)

list_2_no_tags = []
for elem in list_2:
    list_2_no_tags.append((re.sub('_\w+', '', elem[0]), elem[1]))

result = collections.Counter(list_1) & collections.Counter(list_2_no_tags)
intersected_list = list(result.elements())
#print(intersected_list)
#print(len(intersected_list)) #65 интерсекция с учетом частот

list_1_no_freq = [elem[0] for elem in list_1]
list_2_no_freq = [elem[0] for elem in list_2_no_tags]
result = collections.Counter(list_1_no_freq) & collections.Counter(list_2_no_freq)
intersected_list_2 = list(result.elements())
#print(len(intersected_list_2)) #67


stemmer = SnowballStemmer("russian")
stemmed_text_snowball = [stemmer.stem(token) for token in text_tokens_nltk]
#print(' '.join(stemmed_text_snowball))
frequency_distribution_3 = FreqDist(stemmed_text_snowball)
#print(frequency_distribution_3.most_common(100))
#frequency_distribution_3.plot(30, cumulative=False) #примерно тоже самое


nlp_rus = spacy.load('ru_core_news_sm')
doc = nlp_rus(text)
new_text = ' '.join([' '.join([str(token.text) + '_' + str(token.pos_) + '_' + str(token.morph)]) for token in doc])
new_text = re.sub('\|', '_', new_text)
#print(new_text)

texts = []
path = '/Users/yuliakoltsova/Desktop/texts/'
for file in os.listdir(path):
    with open(path + file, encoding='utf-8', errors='ignore') as txt:
        text = txt.read()
        text = text.lower()
        text = re.sub('[^\w+ ]', '', text)
        text = re.sub('[0-9]+', '', text)
        texts.append(text)

texts.pop(0)

labels = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
dic = {'label' : labels, 'text' : texts}
df = pd.DataFrame(dic)
print(df)

corpus = []  # новая версия только с лемматизацией
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    text_lem = mystem.lemmatize(text)
    text_lem = [word for word in text_lem if not word in set(stopwords.words('russian'))]
    text_lem = ' '.join(text_lem)
    corpus.append(text_lem)
    #tagged_list = []
    #for tagged_w in nltk.pos_tag(text_lem, tagset='universal'):
        #tagged_list.append('_'.join(tagged_w))
    #corpus_tagged.append(' '.join(tagged_list))

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() #полученные вектора
y = df.iloc[:, 0].values #метки спама

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
prediction = dict()

classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
prediction["GaussianNB"] = classifier.predict(X_test)
accuracy1 = accuracy_score(y_test,prediction["GaussianNB"])
print(f"Точность модели GaussianNB: {accuracy1}")


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
# Предсказание на мультиномиальном наивном байесовском алгоритме
prediction["MultinomialNB"] = classifier.predict(X_test)
accuracy2 = accuracy_score(y_test,prediction["MultinomialNB"])
print(f"Точность модели MultinomialNB: {accuracy2}")

model = RandomForestClassifier()
model.fit(X_train,y_train)
# Предсказание на решающих деревьях
prediction["RandomForrest"] = classifier.predict(X_test)
accuracy3 = accuracy_score(y_test,prediction["RandomForrest"])
print(f"Точность модели RandomForestClassifier(): {accuracy3}")