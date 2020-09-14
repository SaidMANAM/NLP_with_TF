from tensorflow.keras.preprocessing.text  import Tokenizer
from tensorflow.keras.preprocessing.sequence   import pad_sequences
import json
import numpy as np



with open('/home/said/Téléchargements/sarcasm_dataset/sarcasm.json','r') as f:
    data=json.load(f)

#print(data)#checking if the file is well uploaded
sentences=[]
labels=[]


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
num_epochs = 30
training_size = 20000

for item in data:
    #print("item",item)
    sentences.append(item['headline'])

    labels.append(item['is_sarcastic'])



#print("headline", sentences)
#print("\n")
#print("is_sarcastic", labels)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


tokenizer=Tokenizer(num_words=vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index
training_sentences= tokenizer.texts_to_sequences(training_sentences)
training_padded=pad_sequences(training_sentences)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)