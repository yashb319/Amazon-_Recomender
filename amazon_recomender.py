import numpy as np
import pandas as pd
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import json
#Loading the dataset
df = pd.read_csv(r'C:\Users\Yash\Public\Desktop\input\amazon-fine-food-reviews\Reviews.csv')

print(df.shape)
df.head()
df.shape
df.dtypes
df.info()
df.describe()
#Removing the Duplicates if any
df.duplicated().sum()
df.drop_duplicates(inplace=True)
#Check for the null values in each column
df.isnull().sum()
#Remove the NaN values from the dataset
df.isnull().sum()
df.dropna(how='any',inplace=True)
df.head()
import seaborn as sns
sns.countplot(df['Score'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Score')
rating_df = pd.DataFrame(df, columns=['Score', 'Text'])

print(rating_df.shape)
rating_df.head()
rating_df['Score'].astype('category').value_counts()
dummies = pd.get_dummies(rating_df['Score'])
dummies.head()
#Text Preprocessing
## Lower Casing
rating_df['Text'] = rating_df['Text'].str.lower()
rating_df.head()
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

rating_df['Text'] = rating_df['Text'].apply(lambda text: remove_punctuation(text))
rating_df.head()
## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

rating_df['Text'] = rating_df['Text'].apply(lambda text: remove_stopwords(text))
rating_df.head()
## Removal of urls
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

rating_df['Text'] = rating_df['Text'].apply(lambda text: remove_urls(text))
rating_df.head()
x_train, x_test, y_train, y_test = train_test_split(
    rating_df['Text'],
    dummies,
    test_size=0.1, random_state=19
)


def build_matrix(word_index, path):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(path):
        with open(path) as f:
            embedding_index = {}

            for line in tqdm(f):
                word, arr = get_coefs(*line.strip().split(' '))
                if word in word_index:
                    embedding_index[word] = arr

        return embedding_index

    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in tqdm(word_index.items()):
        try:
            embedding_matrix[i] = embedding_index[word]
            except KeyError:
            pass
    return embedding_matrix


def build_model(embedding_matrix):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = Dense(512, activation='relu')(hidden)

    result = Dense(5, activation='softmax')(hidden)

    model = Model(inputs=words, outputs=result)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
    % % time
    CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'8?÷a•à-ßØ³p‘?´°£€\×™v²—'
    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(list(x_train) + list(x_test))
    embedding_matrix = build_matrix(tokenizer.word_index, 'C:\Users\NEHA GUPTA\Desktop\input\fasttext-crawl-300d-2m\crawl-300d-2M.vec')
    maxlen = 512
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=512)
    x_test = sequence.pad_sequences(x_test, maxlen=512)
    model = build_model(embedding_matrix)
    model.summary()

    checkpoint = ModelCheckpoint(
        'model.h5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto'
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=512,
        callbacks=[checkpoint],
        epochs=10,
        validation_split=0.1
    )
    print(history)
    # Plot training & validation accuracy values
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()