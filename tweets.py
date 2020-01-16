#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import helpers

# tweets
conn = sqlite3.connect('data/db.db')
c = conn.cursor()

with open('data/tweets.txt', 'w', encoding='utf-8') as f:
    # Считываем тексты твитов
    for row in c.execute('SELECT ttext FROM sentiment'):
        if row[0]:
            tweet = helpers.preprocess_text(row[0])
            # Записываем предобработанные твиты в файл
            print(tweet, file=f)
