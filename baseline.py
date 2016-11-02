__author__ = 'lixin77'

import numpy as np
from keras.preprocessing import sequence
from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from evaluate import evaluate
from keras.utils.np_utils import to_categorical
import cPickle
import sys

def ToArray(train, val, test):
    return np.array(train), np.array(val), np.array(test)

def Padding(train, val, test, max_len):
    padded_train = sequence.pad_sequences(train, maxlen=max_len)
    padded_val = sequence.pad_sequences(val, maxlen=max_len)
    padded_test = sequence.pad_sequences(test, maxlen=max_len)
    return padded_train, padded_val, padded_test

stwords = {}
for w in stopwords.words('english'):
    if w not in stwords:
        stwords[w] = 1

def run(model_name, dataset_name):

    """
    run the baseline model
    :param model_name: name of baseline models, CNN or LSTM
    :param dataset_name: name of datasets, candidate values [bbc, digg, MySpace, rw, Twitter, YouTube]
    """
    print "model: %s" % model_name

    print "process dataset %s..." % dataset_name

    data = cPickle.load(open('./pkl/%s.pkl' % dataset_name, 'rb'))

    records, glove_embeddings, vocab, word_to_df = data

    dim_emb = len(glove_embeddings[1])

    # index of word starts from 1
    # initial weights of embedding layer
    embeddings = np.zeros((len(vocab) + 1, dim_emb), dtype='float32')

    for w in vocab:
        wid = vocab[w]
        embeddings[wid, :] = glove_embeddings[wid]

    train_x, train_y, val_x, val_y, test_x, test_y, test_strength = [], [], [], [], [], [], []

    max_len = 0

    for r in records:
        text = r['text']
        y = r['label']
        wids = [vocab[w] for w in text.split(' ')]
        if len(wids) > max_len:
            max_len = len(wids)
        if r['type'] == 'train':
            train_x.append(wids)
            train_y.append(y)
        elif r['type'] == 'val':
            val_x.append(wids)
            val_y.append(y)
        elif r['type'] == 'test':
            strength = r['strength']
            test_x.append(wids)
            test_y.append(y)
            test_strength.append(strength)

    train_x, val_x, test_x = ToArray(train=train_x, val=val_x, test=test_x)
    train_y, val_y, test_y = ToArray(train=train_y, val=val_y, test=test_y)
    _, _, test_strength = ToArray(train=[], val=[], test=test_strength)
    #print train_x.shape, val_x.shape, test_x.shape
    train_x, val_x, test_x = Padding(train=train_x, val=val_x, test=test_x, max_len=max_len)

    batch_size = 50 if model_name == 'CNN' else 32
    if train_x.shape[0] % batch_size:
        n_extra = batch_size - train_x.shape[0] % batch_size
        x_extra = train_x[:n_extra, :]
        y_extra = train_y[:n_extra]
        train_x = np.append(train_x, x_extra, axis=0)
        train_y = np.append(train_y, y_extra, axis=0)
    np.random.seed(38438)
    # shuffle the training set
    train_set = np.random.permutation(zip(train_x, train_y))
    train_x, train_y = [], []
    for (x, y) in train_set:
        train_x.append(x)
        train_y.append(y)

    n_labels = 2

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_y = to_categorical(train_y)
    val_y = to_categorical(val_y)
    print "n_train: %s, n_val: %s, n_test: %s" % (train_x.shape[0], 
        val_x.shape[0], test_x.shape[0])
    
    if model_name == 'CNN':
        model = Graph()
        model.add_input(name='input', input_shape=(max_len,), dtype='int')
        model.add_node(Embedding(input_dim=len(vocab) + 1,
                                 output_dim=dim_emb, input_length=max_len,
                                 weights=[embeddings]), name="emb", input="input")
        filter_hs = [3, 4, 5]
        n_filter = 100
        dropout_rate = 0.5
        n_epoch = 20

        for i in xrange(len(filter_hs)):
            win_size = filter_hs[i]
            conv_name = 'conv%s' % i
            pool_name = "pool%s" % i
            flatten_name = "flatten%s" % i
            pool_size = max_len - win_size + 1
            model.add_node(layer=Convolution1D(nb_filter=n_filter, filter_length=win_size,
                                               activation='relu', W_constraint=maxnorm(m=3),
                                               b_constraint=maxnorm(m=3)),
                           name=conv_name, input='emb',)
            model.add_node(layer=MaxPooling1D(pool_length=pool_size), name=pool_name, input=conv_name)
            model.add_node(layer=Flatten(), name=flatten_name, input=pool_name)
        model.add_node(layer=Dropout(p=dropout_rate), name="dropout", inputs=["flatten0", "flatten1", "flatten2"])
        model.add_node(layer=Dense(output_dim=n_labels, activation='softmax'), name='softmax', input='dropout')

        model.add_output(input='softmax', name="output")

        model.compile(loss={'output':'categorical_crossentropy'}, 
            optimizer='adadelta', metrics=['accuracy'])

        model_path = './model/%s_%s.hdf5' % (model_name, dataset_name)
        best_model = ModelCheckpoint(filepath=model_path, monitor='val_acc',
                                     save_best_only=True, mode='max')
        print "training..."
        model.fit(data={'input': train_x, 'output': train_y}, batch_size=batch_size,
                  nb_epoch=n_epoch, validation_data={'input': val_x, 'output': val_y}, 
                  callbacks=[best_model], verbose=0)

    else:
        model = Sequential()
        model.add(Embedding(input_dim=len(vocab) + 1, output_dim=dim_emb,
                            mask_zero=True, input_length=max_len,
                            weights=[embeddings]))
        model.add(LSTM(output_dim=128,
                    dropout_W=0.2,
                    dropout_U=0.2))

        model.add(Dense(n_labels, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_path = './model/%s_%s.hdf5' % (model_name, dataset_name)
        best_model = ModelCheckpoint(filepath=model_path, monitor='val_acc',
                                     save_best_only=True, mode='max')
        n_epoch = 20
        print "training..."
        model.fit(x=train_x, y=train_y,
                    batch_size=batch_size, nb_epoch=n_epoch,
                    validation_data=(val_x, val_y),
                    callbacks=[best_model])
    pred_strength = []
    print "load the best model from disk..."
    model.load_weights(model_path)
    if model_name == 'LSTM':
        pred_strength = model.predict(x=test_x, batch_size=batch_size)
    else:
        for i in xrange(len(test_x)):
            res = model.predict(data={'input': test_x[i: i+1]}, batch_size=1)
            pred_strength.append(res['output'])
        pred_strength = np.array(pred_strength)
        pred_strength = pred_strength.reshape((pred_strength.shape[0], pred_strength.shape[2]))
        assert pred_strength.shape == test_strength.shape
    print "evaluate performance of the system..."
    accu, ap, rmse = evaluate(strength_gold=test_strength, strength_pred=pred_strength)
    print "%s over %s--->accuracy: %s, ap: %s, rmse: %s\n\n" % (model_name, dataset_name, accu, ap, rmse)

    pred_strengths_lines = []
    for strength in pred_strength:
        pred_strengths_lines.append('%s\n' % ' '.join([str(ele) for ele in strength]))
    with open('./res/%s_pred.txt' % dataset_name, 'w+') as fp:
        fp.writelines(pred_strengths_lines)


if __name__ == '__main__':
    model, dataset = sys.argv[1:]
    run(model_name=model, dataset_name=dataset)



