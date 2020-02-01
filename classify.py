import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plot


# read training dataset
# this is a tab '\t' separated text file containing text and label
training_file_path = './amazon_cells_labelled.txt'
df = pd.read_csv(training_file_path, names=['text', 'label'], sep='\t')

texts = df['text'].to_numpy()
labels = df['label'].to_numpy()


# test_size: optional - proportion of the dataset to include in the test split - .25 is the default
# ramdon_state: optional - the seed used by the random number generator - 359 is my lucky number
# return: List containing train-test split of inputs.
train_x, test_x, train_Y, test_Y = train_test_split(texts, labels, test_size=0.25, random_state=359)

# convert a collection of text documents to a matrix of token counts
count_vectorizer = CountVectorizer()
count_vectorizer.fit(train_x)
train_X = count_vectorizer.transform(train_x)
test_X  = count_vectorizer.transform(test_x)


# define the learning model - more info @ https://keras.io/getting-started/sequential-model-guide/
# you can try vary the first layer dimension and see the result
# the final layer should be 1 since is a binary classification
model = Sequential()
model.add(layers.Dense(16, input_dim=train_X.shape[1], activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

# compile and train the model
# binary_crossentropy is for binary classification and for multi-class dimension use categorical_crossentropy
# try change the optimizer from rmsprop to adam and compare the result
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

result = model.fit(train_X, train_Y,
                   epochs=100,
                   validation_data=(test_X, test_Y),
                   batch_size=10,
                   verbose=False)

# print the result
loss, accuracy = model.evaluate(train_X, train_Y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_X, test_Y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot the result
plot.style.use('ggplot')
plot.figure(figsize=(13.0, 7.0))

x_range = range(1, len(result.history['accuracy']) + 1)

plot.subplot(1, 2, 1)
plot.title('Accuracy')
plot.plot(x_range, result.history['accuracy'], 'b', label='Training')
plot.plot(x_range, result.history['val_accuracy'], 'r', label='Validation')
plot.legend()

plot.subplot(1, 2, 2)
plot.title('Loss')
plot.plot(x_range, result.history['loss'], 'b', label='Training')
plot.plot(x_range, result.history['val_loss'], 'r', label='Validation')
plot.legend()

plot.show()
