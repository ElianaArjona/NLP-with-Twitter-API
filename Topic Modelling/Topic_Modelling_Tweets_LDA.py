import tweepy
import pandas as pd
import gensim
import spacy
import re

from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary

def removeSpecialChar(sentence):

    remove_alfa = ['ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü']

    for i in range(len(remove_alfa)):
        if i == 0:
            sentence = re.sub(remove_alfa[i], 'n', sentence)
        elif i == 1:
            sentence = re.sub(remove_alfa[i], 'a', sentence)
        elif i == 2:
            sentence = re.sub(remove_alfa[i], 'e', sentence)
        elif i == 3:
            sentence = re.sub(remove_alfa[i], 'i', sentence)
        elif i == 4:
            sentence = re.sub(remove_alfa[i], 'o', sentence)
        elif i == 5 or i == 6:
            sentence = re.sub(remove_alfa[i], 'u', sentence)

    return sentence

def removeEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


consumer_key = ""
consumer_secret = ""
access_token = ""
access_secret = ""

#Connet to Tweeter
# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)
twt = []
id = 0


#
# Get Tweets
for tweets in tweepy.Cursor(api.user_timeline, screen_name='@TReporta', tweet_mode="extended").items(300):
    twt.append([id,tweets.full_text])
    id+=1

df = pd.DataFrame(data=twt,columns=['ID','Tweet'])
df.to_csv("./Topic Modelling/tweets.csv")

#Read Tweets
df = pd.read_csv('./Topic Modelling/tweets.csv')

txt = []
for i in df['Tweet']:
    i = removeSpecialChar(i)
    i = removeEmojify(i)
    i = i.replace('\n',' ')
    i = re.sub('\B(#[a-zA-Z0-9]+)(?!;)','',i)
    i = re.sub('https?://(([a-zA-Z]+)(/)*(.)*([a-zA-Z]*))','',i)
    txt.append(i.lower().strip())

print(txt)
text = '\n'.join(txt)

#Process data with Spacy
nlp = spacy.load("./es_core_news_lg")

#Add Stopwords
my_stop_words = ['registrese', 'registrar','probar','pruebelo','probarlo','y',
                 'a',' ','o','u','e','ha','he','gbm','aqui','panamenos','rt_@minsapma',
                 '@bcbrp','rt','️','n_°','panama','  ']

for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

doc = nlp(text)
print(doc)


#Clean Data
#we add some words to the stop word list
texts, article, skl_texts = [], [], []
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        article.append(w.lemma_)
    # if it's a new line, it means we're onto our next document
    if w.text == '\n':
        skl_texts.append(' '.join(article))
        texts.append(article)
        article = []

#Create Bigrams
bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]

#create Bag of Words
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA models
ldamodel = LdaModel(corpus=corpus, num_topics=20, id2word=dictionary)
result = ldamodel.show_topics(num_words=8,num_topics=20)


#Get Results
for r in result:
    print(r)

print(len(result))

