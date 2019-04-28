import json
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import tensorflow as tf
from keras import models
from keras import layers
import numpy as np

data = []
for line in open('Sarcasm_Headlines_Dataset.json', 'r'):
    data.append(json.loads(line))
titles = []
y_vals = []

for i in range(0,len(data)):
    titles.append(data[i]['headline'])
    y_vals.append(data[i]['is_sarcastic'])

# nltk.download('punkt')

titles_tokenized = []
for title in titles:
    titles_tokenized.append(word_tokenize(title))
titles_an = [] # alphanumeric
for title in titles_tokenized:
    words = [word for word in title if word.isalpha()]
    titles_an.append(words)
# Let's now stem the words
porter = PorterStemmer()
titles_preprocessed = []
for title in titles_an:
    stemmed = [porter.stem(word) for word in title]
    titles_preprocessed.append(stemmed)
# Now, let's create a large list of all of the words and find the 10,000 most frequent ones
word_list = []

for title in titles_preprocessed:
    for word in title:
        word_list.append(word)

freq_list = Counter(word_list)
dictionary = freq_list.most_common(10000)
dictionary = list(zip(*dictionary))[0]

#      We now have a list with the 10000 most common words. Let us convert our sentences to lists of these words in
# order to feed it into the Neural Network.
nums = range(0,10000)
word_int = dict(zip(dictionary, nums))
x_vals = []

for title in titles_preprocessed:
    x_vals.append([word_int[x] for x in title if x in word_int.keys()])
# Now, let's format the data for the Neural Network and divide the training, validation, and test sets

x = np.array(x_vals)
test_data = x[:5000]
train_data = x[5000:]


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y = np.asarray(y_vals).astype('float32')
y_test = y[:5000]
y_train = y[5000:]
# Create the validation set
x_val = x_train[:5000]
x_partial_train = x_train[5000:]

y_val = y_train[:5000]
y_partial_train = y_train[5000:]

# Prevent Tensorflow from allocating my entire GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_partial_train, y_partial_train, epochs = 20, batch_size = 512, validation_data=(x_val, y_val))

# Let us train the model with 6 epochs.
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 4, batch_size = 512)

results = model.evaluate(x_test, y_test)

print(results)