from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('chatbot_model.h5')
tokenizer = ...  # Load the tokenizer used during training

maxlen_questions = ...  # Set the maximum length of questions
maxlen_answers = ...  # Set the maximum length of answers

def str_to_tokens(sentence):
    words = sentence.lower().split()
    tokens_list = [tokenizer.word_index[word] for word in words]
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    states_values = model.layers[2].predict(str_to_tokens(user_input))

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''

    while not stop_condition:
        dec_outputs, h, c = model.layers[3].predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None

        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    return jsonify({'response': decoded_translation.strip()})

if __name__ == '__main__':
    app.run(debug=True)
