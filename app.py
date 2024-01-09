import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load conversation data
dir_path = 'chatbot_nlp/data'
files_list = os.listdir(dir_path)

questions = []
answers = []

for filepath in files_list:
    stream = open(os.path.join(dir_path, filepath), 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 1:
            questions.append(con[0])
            answers.append(' '.join(con[1:]))

# Tokenize questions and answers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

vocab_size = len(tokenizer.word_index) + 1
maxlen_questions = max(len(q.split()) for q in questions)
maxlen_answers = max(len(a.split()) for a in answers)

tokenized_questions = tokenizer.texts_to_sequences(questions)
tokenized_answers = tokenizer.texts_to_sequences(answers)

padded_questions = pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
padded_answers = pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')

# Split the data into training and validation sets
encoder_input_data, val_encoder_input_data, decoder_input_data, val_decoder_input_data = train_test_split(
    padded_questions, padded_answers, test_size=0.1, random_state=42
)

# Create one-hot encoded labels for decoder output
onehot_answers = to_categorical(val_decoder_input_data, vocab_size)

# Ensure "start" token is included in decoder input
decoder_input_data = np.concatenate([np.ones((decoder_input_data.shape[0], 1)), decoder_input_data], axis=1)

# Define the model
embedding_dim = 200
latent_dim = 200

# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers + 1,))  # +1 for "start" token
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

# Train the model
batch_size = 50
epochs = 150

# Ensure both inputs have the same number of samples
min_samples = min(encoder_input_data.shape[0], decoder_input_data.shape[0], onehot_answers.shape[0])

model.fit(
    [encoder_input_data[:min_samples], decoder_input_data[:min_samples]],
    onehot_answers[:min_samples],
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# Save the model
model.save('chatbot_model.h5')
