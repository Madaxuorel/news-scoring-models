
import pandas as pd
import contractions
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import re
import nltk
import tqdm
import unicodedata
import contractions
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import load_model

import pickle



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def removeHtmlWords(text):
    return re.sub(r'https:\/\/[^\s]+', '', text)

def strip_html_tags(text):
  """
  Removes html tags from text taken for html pages
  """
  soup = BeautifulSoup(text, "html.parser")
  [s.extract() for s in soup(['iframe', 'script'])]
  stripped_text = soup.get_text()
  stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
  return stripped_text

def remove_accented_chars(text):
  """
  Removes accents from chars
  """
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text

def stopwords_removal(words):
    """
    Remvoves stopwords (english language) from the text
    """
    list_stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in words if word not in list_stopwords]


def pre_process_corpus(docs):
  norm_docs = []
  for doc in tqdm.tqdm(docs):
    #case folding
    doc = doc.lower()
    #remove special characters\whitespaces
    doc = strip_html_tags(doc)
    doc = removeHtmlWords(doc)
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()
    #tokenize
    doc = word_tokenize(doc)
    #filtering
    doc = stopwords_removal(doc)
    norm_docs.append(doc)
  
  norm_docs = [" ".join(word) for word in norm_docs]
  return norm_docs



def newsLineToScore(news,model):
  """
  Array of news lines to array of sentiment scores
  """
  processedNewsLine = pre_process_corpus(news)
  print(processedNewsLine)
  with open('../cnn-models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  
  tokenizedNewsLine = tokenizer.texts_to_sequences(processedNewsLine)
  print(tokenizedNewsLine)
  MAX_LENGTH = 47
  tokenizedNewsLine_padded = pad_sequences(tokenizedNewsLine, maxlen=MAX_LENGTH, padding="post")
  if model == 'CNN':
    model = load_model("../cnn-models/sentimentModel.keras")
  elif model == 'LSTM':
    model = load_model("../lstm-models/sentimentModel_lstm.keras")
  elif model == 'LSTM-word2vec':
    model = load_model("../word2vec-lstm-models/sentimentModel_word2vec_lstm.keras")
      
  return  list(pd.DataFrame(model.predict(tokenizedNewsLine_padded)).apply(lambda x:-1*x[0]+ 1*x[2],axis=1))