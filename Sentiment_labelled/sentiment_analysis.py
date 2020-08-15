import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
import re

""" Load our dataset """
df_yelp = pd.read_table('./Sentiment_labelled/yelp_labelled.txt')
df_imdb = pd.read_table('./Sentiment_labelled/imdb_labelled.txt')
df_amz = pd.read_table('./Sentiment_labelled/amazon_cells_labelled.txt')

# Concatenate our Datasets
frames = [df_yelp,df_imdb,df_amz]

# Renaming Column Headers
for colname in frames:
    colname.columns = ["Message","Target"]

# Column names
for colname in frames:
    print(colname.columns)

# Assign a Key to Make it Easier
keys = ['Yelp','IMDB','Amazon']

# Merge or Concat our Datasets
df = pd.concat(frames,keys=keys)

## Data Cleaning
df = pd.read_csv("./Sentiment_labelled/sentimentdataset.csv")

#Look for Null Values
df.isnull().sum()

"""                      SPACY - Machine Learning With SKlearn               """


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics





nlp = spacy.load('en_core_web_sm-2.3.0')
stop_words= list(STOP_WORDS)

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()
punctuations = string.punctuation


def cleanData(data):
    for i in range(len(data['Message'])):
        data.iloc[i,2] = re.sub('[0-9]', '', str(data.iloc[i,2]))
        data.iloc[i, 2] = re.sub('[.]*[/]*', '', str(data.iloc[i,2]))

    data.dropna(subset=['Message'], inplace=True)
    return data


#Add Stopwords
my_stop_words = ['z','0/10','o','pg-','th','vc','razr','lg','env','lg','dr','s','r','--','oy']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

#Bag of Word
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))

#Term Frequency-Inverse Document Frequency
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

# Split Data
df = cleanData(df)

X = df['Message'] # the features we want to analyze
ylabels = df['Target'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

#Create Model
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)


# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

# Example
sample = ['Very good restaurant','Hate this movie']
print(pipe.predict(sample))

