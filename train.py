import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Step 1: Preprocess the data
data = pd.read_csv('training_data.csv')

# Tokenize the sentences and create vocabulary
tokens = data['Token'].unique()
word2idx = {token: idx + 1 for idx, token in enumerate(tokens)}  # Assign an index to each token
word2idx['<PAD>'] = 0  # Padding token
idx2word = {idx: token for token, idx in word2idx.items()}
vocab_size = len(word2idx)

# Convert tags into numerical labels
tags = data['Tag'].unique()
tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
num_tags = len(tag2idx)

# Step 2: Prepare the training data
sentences = []
labels = []

for record_number, group in data.groupby('Record Number'):
    sentence = [word2idx[token] for token in group['Token']]
    label = [tag2idx[tag] for tag in group['Tag']]
    sentences.append(sentence)
    labels.append(label)

# Pad sequences to the same length
max_length = max(len(sentence) for sentence in sentences)
padded_sentences = pad_sequences(sentences, maxlen=max_length, padding='post')
padded_labels = pad_sequences(labels, maxlen=max_length, padding='post')

# Convert labels to one-hot encoding
one_hot_labels = to_categorical(padded_labels, num_classes=num_tags)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sentences, one_hot_labels, test_size=0.2)

# Step 3: Design the model architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dense(units=num_tags, activation='softmax'))

# Step 4: Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')
