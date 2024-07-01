import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Loading dataset using pandas
df = pd.read_csv('IMDB Dataset.csv')

# Displaying the first few rows of the dataframe
print(df.head())


# Cleaning and preprocessing the data
def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove HTML tags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenizing text
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)  # Join words into a single string with spaces


# Apply function to the review column
df['review'] = df['review'].apply(preprocess_text)

# Displaying result of processing
print(df.head())

# Tokenization and Padding
max_words = 10000  # maximum words to be used in the tokenization
max_len = 150  # maximum length of sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['review'])  # mapping words to integers
sequences = tokenizer.texts_to_sequences(df['review'])  # converts review to sequence
padded_sequences = pad_sequences(sequences, maxlen=max_len)  # pads or truncates sequence to ensure same length

# Label encoding
labels = pd.get_dummies(df['sentiment']).values  # one-hot encoded labels

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Developing the model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Predicting
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Classification report
class_report = classification_report(y_true, y_pred, target_names=['negative', 'positive'])
print("Classification Report:")
print(class_report)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
