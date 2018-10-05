# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy
import spaCy

# A dictionary mapping words to an integer index
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
def encode_review(string):
    w, h = 256, 1;
    Matrix = [[0 for x in range(w)] for y in range(h)] 
    Matrix[0][0] = 1
    i = 1
    for x in string.split(' '):
       if(word_index.get(x)!=None):
           Matrix[0][i]=word_index[x]
       else:
           Matrix[0][i]= (2)
       i+=1
    Matrix[0] = keras.preprocessing.sequence.pad_sequences([Matrix[0]],value = word_index["<PAD>"],padding='post',maxlen=256)
    m = Matrix[0];
    #rez = [[m[j][i] for j in range(len(m))] for i in range(1)]     
    return(m)
if __name__=="__main__":
    print(tf.__version__)
    
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=256)
    vocab_size = 20000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(16, activation=tf.nn.tanh))
    model.add(keras.layers.Dense(16, activation=tf.nn.softmax))
    model.add(keras.layers.Dense(16, activation=tf.nn.tanh))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer=tf.train.AdamOptimizer(),loss='binary_crossentropy',metrics=['accuracy'])
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=50,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
    results = model.evaluate(test_data, test_labels)
    print(results)
    #print(decode_review(test_data[0]))
    history_dict = history.history
    history_dict.keys()

    dict_keys = (['val_loss', 'loss', 'val_acc', 'acc'])
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    print("------------")
    results = model.evaluate(test_data, test_labels)
    print(results)
    print("--------Checking against user input-------")
    while True:
        xyz = input("Enter String: ")
        if(xyz.lower()==str('break')):
            break
        xyz1 = encode_review(xyz)
        xyz2 = model.predict(x=xyz1)
        xyz22 = model.predict_classes(x=xyz1)
        print('---------')
        print(xyz2)
        print(xyz22)





    