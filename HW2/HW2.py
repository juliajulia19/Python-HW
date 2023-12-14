import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
import re

all_page = ['https://sysblok.ru', ]

for i in range(2, 17):
    link = 'https://sysblok.ru/' + 'page/' + str(i)
    all_page.append(link)

all_links = []

for i in all_page[:2]: #убрать позже ограничения
    url = i
    page = rq.get(url)
    soup = BeautifulSoup(page.text, features="html.parser")
    for link in soup.find_all("a"):
        if link.parent.name == 'h2':
            all_links.append(link.get('href'))

# для парсинга даты: извлеките текст из тега time
url0 = all_links[0]
page0 = rq.get(url0)
soup0 = BeautifulSoup(page0.text, features="html.parser")

date0 = soup0.find('time').text

# для парсинга заголовка: извлеките текст из тега h1
header0 = soup0.find('h1').text

# для парсинга текста: соберите все тексты из тега p и соедините в строку
text0 = []
for i in soup0.find_all('p'):
  if i.parent.name == 'article':
    text0.append(i.text.strip())
final_text0 = ' '.join(text0)
print(final_text0)

# автора собрать сложнее:
author0 = soup0.find('meta', {'name' : 'author'}).attrs['content']

# тематические категории:
categories0 = []
for i in soup0.find_all('a'):
    if i.get('rel') == ['category', 'tag']:
        categories0.append(i.text)


def GetNews(url):
  page = rq.get(url)
  soup = BeautifulSoup(page.text, features="html.parser")
  author = soup.find('meta', {'name' : 'author'}).attrs['content']
  date = soup.find('time').text
  header = soup.find('h1').text
  text = []
  for i in soup.find_all('p'):
      if i.parent.name == 'article':
          text.append(i.text.strip())
  final_text = ' '.join(text)
  categories = []
  for i in soup.find_all('a'):
      if i.get('rel') == ['category', 'tag']:
          categories.append(i.text)
  return url, author, categories, date, header, final_text

news = [] # список с новостями
for link in all_links[:3]: #убрать ограничения
    try:
        new = GetNews(link)
        news.append(new)
    except:
        print(link)


# соберите всю собранную информацию в датафрейм
df = pd.DataFrame(news)
df.columns = ['link', 'author', 'categories', 'date', 'title', 'text']
print(df.head())



