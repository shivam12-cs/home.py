# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Step 2: Data Cleaning and Preprocessing
train_data = train_data.fillna('')
test_data = test_data.fillna('')

# Step 3: Text Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])

X = tokenizer.texts_to_sequences(train_data['text'])
X = pad_sequences(X)
Y = train_data['target']

# Step 4: Model Architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Split the data for training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 6: Train the model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=64)

# Step 7: Make predictions on the test data
test_sequences = tokenizer.texts_to_sequences(test_data['text'])
test_sequences = pad_sequences(test_sequences)
predictions = model.predict(test_sequences)
predictions = (predictions > 0.5).astype(int)

# Step 8: Create a submission file for Kaggle
submission = pd.DataFrame({'id': test_data['id'], 'target': predictions.ravel()})
submission.to_csv('submission.csv', index=False)

# Step 9: Submit your results on Kaggle and get your leaderboard position

