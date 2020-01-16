import re
import pandas as pd

import numpy as np
from gensim.models import Word2Vec
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, concatenate, Activation, Dropout
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split

# Matrix weight (max word count in tweets)
SENTENCE_LENGTH = 26

# Dictionary size
NUM = 100000

# Load model
w2v_model = Word2Vec.load('models/w2v/model.w2v')
DIM = w2v_model.vector_size


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)

    return text.strip()

# Since Keras 2.0 metrics F-measure, precision, and recall have been removed,
# so the following code was found in the history of the repo.


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        return true_positives / (predicted_positives + K.epsilon())

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_tokenizer(train_texts):
    tokenizer = Tokenizer(num_words=NUM)
    tokenizer.fit_on_texts(train_texts)

    return tokenizer


def get_model(tokenizer, show_summary=True):
    # initialize zero matrix of embedding layer
    embedding_matrix = np.zeros((NUM, DIM))
    # Dodajemy NUM=100000 najczesciej spotykanych słow z nauki World2Vec w warstwę embedding
    for word, i in tokenizer.word_index.items():
        if i >= NUM:
            break
        if word in w2v_model.wv.vocab.keys():
            embedding_matrix[i] = w2v_model.wv[word]
    tweet_input = Input(shape=(SENTENCE_LENGTH,), dtype='int32')
    tweet_encoder = Embedding(NUM, DIM, input_length=SENTENCE_LENGTH,
                              weights=[embedding_matrix], trainable=False)(tweet_input)

    branches = []
    # Dropout regularyzacja porzucania z prawdopodobieństwem upuszczenia wysokości p=0.2
    # Pomaga dla przekwalifikowania modelu
    x = Dropout(0.2)(tweet_encoder)

    # dodanie filtrów
    for size, filters_count in [(2, 10), (3, 10), (4, 10), (5, 10)]:
        for i in range(filters_count):
            # dodajemy warstwy konwolucyjnej
            branch = Conv1D(filters=1, kernel_size=size, padding='valid', activation='relu')(x)
            # próbkowanie w dół
            branch = GlobalMaxPooling1D()(branch)
            branches.append(branch)

    # Konkatynacja mapy objektów
    x = concatenate(branches, axis=1)

    # Dropout regularyzacja
    x = Dropout(0.2)(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    model = Model(inputs=[tweet_input], outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall, f1])
    if show_summary:
        model.summary()

    return model


def get_positive_negative_tables():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']

    return pd.read_csv('data/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text']),\
           pd.read_csv('data/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])


def get_train_test_data():
    data_positive, data_negative = get_positive_negative_tables()

    sample_size = min(data_positive.shape[0], data_negative.shape[0])
    labels = [1] * sample_size + [0] * sample_size

    raw_data = np.concatenate(
        (data_positive['text'].values[:sample_size], data_negative['text'].values[:sample_size]),
        axis=0
    )

    return train_test_split([preprocess_text(t) for t in raw_data], labels, test_size=0.2, random_state=1)
