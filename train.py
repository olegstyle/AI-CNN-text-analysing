#! /usr/bin/env python
# -*- coding: utf-8 -*-

import glob

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

from helpers import *

# get data
x_train, x_test, y_train, y_test = get_train_test_data()

# create and learn Tokenizer
tokenizer = get_tokenizer(x_train)
# Map each text to an array of token identifiers
x_train_seq = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=SENTENCE_LENGTH)
x_test_seq = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=SENTENCE_LENGTH)

model = get_model(tokenizer)
checkpoint = ModelCheckpoint("models/cnn/cnn-frozen-embeddings-{epoch:02d}-{val_f1:.2f}.hdf5", monitor='val_f1',
                             save_best_only=True, mode='max', period=1)
history = model.fit(x_train_seq, y_train, batch_size=32, epochs=10, validation_split=0.25, callbacks=[checkpoint])

# Выбираем лучший вес модели (выбрал модель с наивысшими показателями F-меры на валидационном наборе данных)
cnnList = glob.glob('models/cnn/*.hdf5')
bestCnnFile = cnnList[0]
if len(bestCnnFile) > 0:
    for cnnFile in cnnList:
        print(bestCnnFile.split('-')[-1])
        a = bestCnnFile.split('-')[-1].replace('.hdf5', '')
        b = cnnFile.split('-')[-1].replace('.hdf5', '')
        if float(a) < float(b):
            bestCnnFile = cnnFile

# Загружаем веса модели
model.load_weights(bestCnnFile)
# Делаем embedding слой способным к обучению
model.layers[1].trainable = True
# Уменьшаем learning rate
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[precision, recall, f1])
model.summary()

checkpoint = ModelCheckpoint("models/cnn/cnn-trainable-{epoch:02d}-{val_f1:.2f}.hdf5", monitor='val_f1',
                             save_best_only=True, mode='max', period=1)
history_trainable = model.fit(x_train_seq, y_train, batch_size=32, epochs=5, validation_split=0.25,
                              callbacks=[checkpoint])
