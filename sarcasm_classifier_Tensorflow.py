from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99 and logs.get('val_accuracy') > 0.90):
            self.model.stop_training = True

# importing the json file for the dataset

with open('/home/said/Téléchargements/sarcasm_dataset/sarcasm.json', 'r') as f:
    data = json.load(f)

#instanciating objects and creating variables
callback=myCallback()
sentences = []
labels = []
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
num_epochs = 50
training_size = 20000

for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# Splitting data into training_validation sets
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# tokenizing and padding the sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sentences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sentences)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

def prediction( mymodel, test,  labels):
    pred=mymodel.predict(test)
    mae = tf.keras.losses.MeanAbsoluteError()
    mae(pred,labels)
    i=0
    for elt in pred:
        if (elt>0.5) :
            print("the sentence number {} is sarcastic".format(i))
        else:
            print("the sentence number {} is not sarcastic".format(i))
    i=i+1

    return pred,mae

sentence=["France won the world cup in 2018",
          "Forecasters call for weather on Monday",
          "Cows lose their jobs as milk prices drop"]
t=[0,1,1]
sequences=tokenizer.texts_to_sequences(sentence)
padded=pad_sequences(sequences)
predi=prediction( model, padded,  t)

