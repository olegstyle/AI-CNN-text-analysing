from helpers import *
from keras.preprocessing.sequence import pad_sequences
import sys

texts = []
for text in sys.argv[1:]:
    texts.append(preprocess_text(text))

if len(texts) == 0:
    print('Nothing to do!')
    exit(0)
print('texts: ', texts)

print('Create and learn Tokenizer')
tokenizer = get_tokenizer(get_train_test_data()[0])
print('Tokenizer learned')

print('Load model')
model = get_model(tokenizer=tokenizer, show_summary=False)
model.load_weights('models/cnn/cnn-trainable-01-0.00.hdf5')
model.summary()
print('Model loaded')

prediction = model.predict(pad_sequences(
    tokenizer.texts_to_sequences(texts),
    maxlen=SENTENCE_LENGTH
))
print('predictions:')
print(prediction)

