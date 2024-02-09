import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
from collections import Counter, defaultdict
from pymystem3 import Mystem
import spacy
from nltk.collocations import *
import itertools
from razdel import sentenize
from razdel import tokenize as razdel_tokenize



#задание 1

# text2 = 'Прекрасный летний лень. Прелесть как дивно видеть расцвет природы, красоту её возрождения. Петров тем временем шёл за прекрасной работы часами'
#
# result = re.findall("[Пп]рекрас[a-яё]+", text2) #подойдет для данной строки
# result2 = re.findall("[a-яA-ЯЁё]*[Пп]рекрас[a-яё]+", text2) #более универсальная регулярка на случай, например, "распрекрасный"
# print(result)

#задание 2

lst_emails = ['student@edumail.hse.ru', 'apetrov@gmail.com', 'msidorova@ya.ru']
def domens_list(list):
    domens = re.findall("@[A-Za-za-яA-ЯЁё/.-]+", ' '.join(list)) #соединение листа в строку добавила, тк для примера был дан список.Если бы была строка, то соединять не нужно было бы - убрать join и передавать просто строку
    domens = re.sub('@', '', ' '.join(domens)) #убираем @, чтобы ответ соответствовал примеру в задании
    domens = domens.split() #превращаем строку в список
    return domens

#print(domens_list(lst_emails))

#задание 3

text = '''Впрочем, сам виноват, кругом виноват! Не пускаться бы на старости лет с клочком волос в амуры да в экивоки…
И еще скажу, маточка: чуден иногда человек, очень чуден. И, святые вы мои! о чем заговорит, занесет подчас!
А что выходит-то, что следует-то из этого? Да ровно ничего не следует, а выходит такая дрянь,
что убереги меня, господи! Я, маточка, я не сержусь, а так досадно только очень вспоминать обо всем,
досадно, что я вам написал так фигурно и глупо. И в должность-то я пошел сегодня таким гоголем-щеголем;
сияние такое было на сердце. На душе ни с того ни с сего такой праздник был; весело было!
За бумаги принялся рачительно — да что вышло-то потом из этого! Уж потом только как осмотрелся, так всё стало по-прежнему
— и серенько и темненько. Всё те же чернильные пятна, всё те же столы и бумаги, да и я всё такой же;
так, каким был, совершенно таким же и остался, — так чего же тут было на Пегасе-то ездить? Да из чего это вышло-то всё?
Что солнышко проглянуло да небо полазоревело! от этого, что ли? Да и что за ароматы такие, когда на нашем дворе под окнами
и чему-чему не случается быть! Знать, это мне всё сдуру так показалось. А ведь случается же иногда заблудиться так человеку
в собственных чувствах своих да занести околесную. Это ни от чего иного происходит, как от излишней, глупой горячности сердца.
Домой-то я не пришел, а приплелся; ни с того ни с сего голова у меня разболелась; уж это, знать, всё одно к одному. 
(В спину, что ли, надуло мне.) Я весне-то обрадовался, дурак дураком, да в холодной шинели пошел.
И в чувствах-то вы моих ошиблись, родная моя! Излияние-то их совершенно в другую сторону приняли.
Отеческая приязнь одушевляла меня, единственно чистая отеческая приязнь, Варвара Алексеевна;
ибо я занимаю у вас место отца родного, по горькому сиротству вашему; говорю это от души,
от чистого сердца, по-родственному. Уж как бы там ни было, а я вам хоть дальний родной, хоть, по пословице,
и седьмая вода на киселе, а все-таки родственник, и теперь ближайший родственник и покровитель; ибо там,
где вы ближе всего имели право искать покровительства и защиты, нашли вы предательство и обиду.
А насчет стишков скажу я вам, маточка, что неприлично мне на старости лет в составлении стихов упражняться.
Стихи вздор! За стишки и в школах теперь ребятишек секут… вот оно что, родная моя.'''

#сделаем небольшую предобработку
mystem = Mystem()
text = text.lower()
text = re.sub('[^\w+ ]', '', text)  #удаляем пунктуацию
lemmas = mystem.lemmatize(text)
words_clean = []
for word in lemmas:
    if word not in stop_words:  #чистим от стоп-слов
        words_clean.append(word)
frequencies_list = FreqDist(words_clean)
del frequencies_list[" "] #удалила лишние одинарные и двойные пробелы, которые появились после лемматизации
del frequencies_list["  "]
freq_dict = dict((word, freq) for word, freq in frequencies_list.items()) #создадим словарь с частотами, который можно распечатать и посмотреть

#print(frequencies_list.most_common(50))

top_n = 25
labels = [element[0] for element in frequencies_list.most_common(top_n)]
counts = [element[1] for element in frequencies_list.most_common(top_n)]
plt.figure(figsize=(15, 10))
plt.title("Самые частые слова в корпусе")
plt.ylabel("Count")
plt.xlabel("Word")
plt.xticks(rotation=90)
plot = sns.barplot(x=labels, y=counts)
#plt.show()

#на мой взгляд получилось примерное распределение по Ципфу, когда второе по используемости слово встречается примерно в два раза реже, чем первое, третье — в три раза реже, чем первое, и тд.
#но не очень яркий пример, так как корпус маленький и большинство слов встречется 1-2 раза

#задание 4

nlp_rus = spacy.load('ru_core_news_sm')
doc = nlp_rus(text)
new_text = ' '.join([' '.join([str(token.text) + '_' + str(token.pos_)]) for token in doc])

list_of_verbs = re.findall('[\w]+_VERB', (new_text))
list_of_nouns = re.findall('[\w]+_NOUN', (new_text))
list_of_adjectives = re.findall('[\w]+_ADJ', (new_text))
list_of_pronouns = re.findall('[\w]+_PRON', (new_text))
list_of_prepos = re.findall('[\w]+_ADP', (new_text))

pos = ['verbs', 'nouns', 'adjectives', 'pronouns', 'prepos']

quantities = []
quantities.append(len(list_of_verbs))
quantities.append(len(list_of_nouns))
quantities.append(len(list_of_adjectives))
quantities.append(len(list_of_pronouns))
quantities.append(len(list_of_prepos))

shares = []
shares.append(len(list_of_verbs)/len(word_tokenize(text))) #делим количество частей речи на количество всех слов в тексте
shares.append(len(list_of_nouns)/len(word_tokenize(text)))
shares.append(len(list_of_adjectives)/len(word_tokenize(text)))
shares.append(len(list_of_pronouns)/len(word_tokenize(text)))
shares.append(len(list_of_prepos)/len(word_tokenize(text)))

data = []
data.append(pos)
data.append(quantities)
data.append(shares)

df = pd.DataFrame(data).transpose()
df.columns =['pos', 'quantities', 'shares']
#print(df)

#задание 5

#nlp_rus = spacy.load('ru_core_news_sm') эта часть кода закомментирована, тк такие строки есть в предыдущем задании
#doc = nlp_rus(text)
list_of_children = []
for token in doc:
    TokAndCh = (token.text, len([child for child in token.children]))
    list_of_children.append(TokAndCh)

sorted_list = sorted(list_of_children, key=lambda x: int(x[1]), reverse=True) #сортируем список по обуванию количества дочерей
#print(sorted_list[:10]) #выводим топ-10

#задание 6

stops = set(stopwords.words('russian') + ["это", "весь"])

def normalize(text):
    doc = nlp_rus(text)
    list_of_lemmas = ' '.join([tok.lemma_ for tok in doc])
    tokens = re.findall('[а-яёa-z0-9]+', list_of_lemmas.lower())
    normalized_text = [word for word in tokens if len(word) > 2 and word not in stops]

    return normalized_text


def preprocess(text):
    sents = sentenize(text)
    return [normalize(sent.text) for sent in sents]


def score_bigrams(unigrams, bigrams, scorer, threshold=-100000, min_count=1):
    ## посчитаем метрику для каждого нграмма
    bigram2score = Counter()
    len_vocab = len(unigrams)
    for bigram in bigrams:
        score = scorer(unigrams[bigram[0]], unigrams[bigram[1]],
                       bigrams[bigram], len_vocab, min_count)

        ## если метрика выше порога, добавляем в словарик
        if score > threshold:
            bigram2score[bigram] = score

    return bigram2score

with open('PDOprax1.txt', encoding = 'utf-8') as file:
    text3 = file.read()

text3 = preprocess(text3)

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder2 = BigramCollocationFinder.from_documents(text3)
finder3 = TrigramCollocationFinder.from_documents(text3)

finder2.nbest(bigram_measures.likelihood_ratio, 15) #первая метрика LL
scores = finder2.score_ngrams(bigram_measures.raw_freq)
#print(scores)
finder3.nbest(trigram_measures.pmi, 15)  #метрика PMI
scores2 = finder3.score_ngrams(trigram_measures.raw_freq)
#print(scores2)

#задание 7
with open('PDOprax1.txt', encoding = 'utf-8') as file:
    text4 = file.read()
doc2 = nlp_rus(text4)
ents = []
for ent in doc2.ents:
    ents.append((ent.text, ent.label_))
print(ents)
print('В вашем тексте', len(ents), 'именованных сущностей')
lstlabel = [ent.label_ for ent in doc2.ents]
entities_dict = {}
for i in lstlabel:
    if i not in entities_dict:
        entities_dict[i] = 1
    else:
        entities_dict[i] += 1

entities = list(entities_dict.keys())
en_counts = list(entities_dict.values())

df2 = pd.DataFrame({
            'Label': entities,
             'Counts': en_counts}
                )
print(df2)






