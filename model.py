__author__ = 'lixin77'

import numpy as np
from keras.models import Sequential, Graph
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Embedding, Merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import cPickle
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import sequence
from nltk.corpus import stopwords
import theano
import theano.tensor as T
from evaluate import *
from scipy import stats

def KLDivergence(y_true, y_pred):
    """
    training objective in our HCNN
    """
    # a small constant added in the formula to avoid denominator being euqal to 0
    smoother = 0.00001
    results, updates = theano.scan(lambda y_t, y_p:
        (y_t + smoother) * (T.log(y_t + smoother) - T.log(y_p + smoother)),
        sequences=[y_true, y_pred])
    return T.sum(results, axis=-1)

def EuclideanDistance(y_true, y_pred):
    results, updates = theano.scan(lambda y_t, y_p:
        (y_t - y_p) * (y_t - y_p),
        sequences=[y_true, y_pred])
    return T.sum(results, axis=-1)

def ToArray(train, val, test):
    return np.array(train), np.array(val), np.array(test)

def Padding(train, val, test, max_len):
    padded_train = sequence.pad_sequences(train, maxlen=max_len)
    padded_val = sequence.pad_sequences(val, maxlen=max_len)
    padded_test = sequence.pad_sequences(test, maxlen=max_len)
    return padded_train, padded_val, padded_test

def Lookup2D(Array2D, LookupTable, dim):
    X = np.zeros((Array2D.shape[0], Array2D.shape[1], dim))
    for i in xrange(Array2D.shape[0]):
        arr = Array2D[i]
        X[i, :, :] = LookupTable[arr]
    # input shape of the 2D Convolutional layer
    return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

stwords = {}
for w in stopwords.words('english'):
    if w not in stwords:
        stwords[w] = 1

dataset_name = 'digg'

print "process dataset %s..." % dataset_name

data = cPickle.load(open('./pkl/%s.pkl' % (dataset_name), 'rb'))

records, glove_embeddings, vocab, word_to_df = data

# meaningful words in the dataset
meaningful_words = []

df_values = word_to_df.values()
#df_threshold = int(np.percentile(df_values, 85) + 0.5)
df_threshold = 6
#df_threshold = stats.mode(df_values)[0][0]
print "df_threshold is:", df_threshold
if dataset_name == 'Twitter_bbc' or dataset_name == 'bbc_Twitter':
    df_threshold = 10
for w in vocab:
    # when performing cross-testing, just double the df_threshold
    if word_to_df[w] > df_threshold and w not in stwords:
        meaningful_words.append(vocab[w])

print "n_meaningful: %s, n_words: %s" % (len(meaningful_words), len(vocab))
# transform word id to one hot vector representation
onehot_converter = LabelBinarizer()

onehot_converter.fit(meaningful_words)

# embeddings in the experiment, type: numpy array
embeddings = np.zeros((len(vocab) + 1, len(meaningful_words) + len(glove_embeddings[1])))
#embeddings = np.zeros((len(vocab) + 1, len(meaningful_words)))
dim_w = len(embeddings[0])

for w in vocab:
    wid = vocab[w]
    onehot = onehot_converter.transform([wid])[0]
    glove = glove_embeddings[wid]
    embeddings[wid, :] = (np.concatenate((onehot, glove), axis=0))
    # For HCNN-no-onehot
    #embeddings[wid, :] = glove
    # For HCNN-no-glove
    #embeddings[wid, :] = onehot


train_sen, train_y, train_strength, val_sen, val_y, val_strength, test_sen, test_y, test_strength \
    = [], [], [], [], [], [], [], [], []

# words in each group
train_J, train_N, train_V = [], [], []
val_J, val_N, val_V = [], [], []
test_J, test_N, test_V = [], [], []

# max length of each group, e.g, sentence group, verb group..
max_sen, max_J, max_N, max_V = 0, 0, 0, 0

for rec in records:
    text = rec['text']
    strength = rec['strength']
    y = rec['label']
    pos = rec['pos']
    wids = [vocab[w] for w in text.split(' ')]
    assert len(wids) == len(pos)
    id_tag_tuple = zip(pos, wids)
    J, N, V = [], [], []
    for (tag, wid) in id_tag_tuple:
        if tag == 4:
            continue
        elif tag == 3:
            if wid in meaningful_words:
                N.append(wid)
        elif tag == 2:
            if wid in meaningful_words:
                V.append(wid)
        elif tag == 1:
            J.append(wid)
    # word id list after incorporate pos-level information
    #for wid in J + V + N:
    #    wids.append(wid)
    if rec['type'] == 'train':
        train_sen.append(wids)
        train_y.append(y)
        train_strength.append(strength)
        train_J.append(J)
        train_N.append(N)
        train_V.append(V)
    elif rec['type'] == 'val':
        val_sen.append(wids)
        val_y.append(y)
        val_strength.append(strength)
        val_J.append(J)
        val_N.append(N)
        val_V.append(V)
    else:
        test_sen.append(wids)
        test_y.append(y)
        test_strength.append(strength)
        test_J.append(J)
        test_N.append(N)
        test_V.append(V)
    if max_sen < len(wids):
        max_sen = len(wids)
    if max_J < len(J):
        max_J = len(J)
    if max_N < len(N):
        max_N = len(N)
    if max_V < len(V):
        max_V = len(V)
print "max_sen:", max_sen
print "max_J:", max_J
print "max_N:", max_N
print "max_V", max_V

print "n_train: %s, n_val: %s, n_test: %s" % (len(train_sen), len(val_sen), len(test_sen))
train_sen, val_sen, test_sen = ToArray(train=train_sen, val=val_sen, test=test_sen)
train_strength, val_strength, test_strength = ToArray(train=train_strength,
    val=val_strength, test=test_strength)
train_J, val_J, test_J = ToArray(train=train_J, val=val_J, test=test_J)
train_N, val_N, test_N = ToArray(train=train_N, val=val_N, test=test_N)
train_V, val_V, test_V = ToArray(train=train_V, val=val_N, test=test_N)

print "padding each group"
train_sen, val_sen, test_sen = Padding(train=train_sen, val=val_sen,
    test=test_sen, max_len=max_sen)
train_J, val_J, test_J = Padding(train=train_J, val=val_J, test=test_J, max_len=max_J)
train_N, val_N, test_N = Padding(train=train_N, val=val_N, test=test_N, max_len=max_N)
train_V, val_V, test_V = Padding(train=train_V, val=val_V, test=test_V, max_len=max_V)

np.random.seed(38438)
#np.random.seed(11345)

batch_size = 32

# make the training processes can be conducted batch by batch
if train_sen.shape[0] % batch_size:
    n_extra = batch_size - train_sen.shape[0] % batch_size

    extra_sen = train_sen[:n_extra, :]
    extra_strength = train_strength[:n_extra, :]
    extra_J = train_J[:n_extra, :]
    extra_N = train_N[:n_extra, :]
    extra_V = train_V[:n_extra, :]

    train_sen = np.append(train_sen, extra_sen, axis=0)
    train_strength = np.append(train_strength, extra_strength, axis=0)
    train_J = np.append(train_J, extra_J, axis=0)
    train_N = np.append(train_N, extra_N, axis=0)
    train_V = np.append(train_V, extra_V, axis=0)

# shuffle the training set
train_set = np.random.permutation(zip(train_sen, train_strength,
    train_J, train_N, train_V))
train_sen, train_strength = [], []
train_J, train_N, train_V = [], [], []

for (sen, strength, J, N, V) in train_set:
    train_sen.append(sen)
    train_strength.append(strength)
    train_J.append(J)
    train_N.append(N)
    train_V.append(V)

train_sen = np.asarray(train_sen)
train_strength = np.asarray(train_strength)
train_J = np.array(train_J)
train_N = np.array(train_N)
train_V = np.array(train_V)

print "look up embeddings..."
train_X_sen = Lookup2D(Array2D=train_sen, LookupTable=embeddings, dim=dim_w)
val_X_sen = Lookup2D(Array2D=val_sen, LookupTable=embeddings, dim=dim_w)
test_X_sen = Lookup2D(Array2D=test_sen, LookupTable=embeddings, dim=dim_w)

train_X_J = Lookup2D(Array2D=train_J, LookupTable=embeddings, dim=dim_w)
val_X_J = Lookup2D(Array2D=val_J, LookupTable=embeddings, dim=dim_w)
test_X_J = Lookup2D(Array2D=test_J, LookupTable=embeddings, dim=dim_w)

train_X_N = Lookup2D(Array2D=train_N, LookupTable=embeddings, dim=dim_w)
val_X_N = Lookup2D(Array2D=val_N, LookupTable=embeddings, dim=dim_w)
test_X_N = Lookup2D(Array2D=test_N, LookupTable=embeddings, dim=dim_w)

train_X_V = Lookup2D(Array2D=train_V, LookupTable=embeddings, dim=dim_w)
val_X_V = Lookup2D(Array2D=val_V, LookupTable=embeddings, dim=dim_w)
test_X_V = Lookup2D(Array2D=test_V, LookupTable=embeddings, dim=dim_w)

print "build sentence-level model..."
CNN_sen = Sequential()
# hyper-parameter of convolutional layer
n_filter_sen = 80
win_size_sen = 1
CNN_sen.add(Convolution2D(nb_filter=n_filter_sen,
                        nb_row=win_size_sen, nb_col=dim_w,
                        input_shape=(1, max_sen, dim_w), dim_ordering='th',
                        border_mode='valid'))
# non-linear layer
CNN_sen.add(Activation(activation='relu'))
# max-pooling layer
CNN_sen.add(MaxPooling2D(pool_size=(max_sen - win_size_sen + 1, 1)))
CNN_sen.add(Flatten())

print "build pos-level model..."
n_filter_pos = 2
win_size_pos = 1
CNN_J = Sequential()
CNN_J.add(Convolution2D(nb_filter=n_filter_pos,
                        nb_row=win_size_pos, nb_col=dim_w,
                        input_shape=(1, max_J, dim_w), dim_ordering='th',
                        border_mode='valid'))
#CNN_J.add(Activation(activation='relu'))
CNN_J.add(MaxPooling2D(pool_size=(max_J - win_size_pos + 1, 1)))
CNN_J.add(Flatten())

CNN_N = Sequential()
CNN_N.add(Convolution2D(nb_filter=n_filter_pos,
    nb_row=win_size_pos, nb_col=dim_w,
    input_shape=(1, max_N, dim_w), dim_ordering='th',
    border_mode='valid'))
#CNN_N.add(Activation(activation='relu'))
CNN_N.add(MaxPooling2D(pool_size=(max_N - win_size_pos + 1, 1)))
CNN_N.add(Flatten())

CNN_V = Sequential()
CNN_V.add(Convolution2D(nb_filter=n_filter_pos,
    nb_row=win_size_pos, nb_col=dim_w,
    input_shape=(1, max_V, dim_w), dim_ordering='th',
    border_mode='valid'))
#CNN_V.add(Activation(activation='relu'))
CNN_V.add(MaxPooling2D(pool_size=(max_V - win_size_pos + 1, 1)))
CNN_V.add(Flatten())


# The proposed model
HCNN = Sequential()

HCNN.add(Merge([CNN_sen, CNN_J, CNN_N, CNN_V], mode='concat', concat_axis=1))
print "add fully connected layer.."
# default initialization method is glorot uniform
HCNN.add(Dense(100))
HCNN.add(Activation('relu'))
HCNN.add(Dropout(0.5))

print "add softmax layer..."
HCNN.add(Dense(2))

HCNN.add(Activation(activation='softmax'))
print "compile the model"
HCNN.compile(optimizer='adadelta', loss=KLDivergence)

print "training..."

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

model_path = './model/%s.hdf5' % dataset_name
best_model = ModelCheckpoint(filepath=model_path, monitor='val_loss',
    save_best_only=True, mode='min')

HCNN.fit(x=[train_X_sen, train_X_J, train_X_N, train_X_V], y=train_strength,
          batch_size=batch_size,
          nb_epoch=30,
          validation_data=([val_X_sen, val_X_J, val_X_N, val_X_V], val_strength),
          callbacks=[early_stopping, best_model])

print "load the best model from disk..."
HCNN.load_weights(model_path)

pred_strength = HCNN.predict(x=[test_X_sen, test_X_J, test_X_N, test_X_V],
    batch_size=batch_size)

print "evaluate performance of the system..."
accu, ap, rmse = evaluate(strength_gold=test_strength, strength_pred=pred_strength)
print "accuracy: %s, ap: %s, rmse: %s" % (accu, ap, rmse)

"""
# for HCNN-no-pos
CNN_sen.add(Dense(100))
CNN_sen.add(Activation('relu'))
CNN_sen.add(Dropout(0.5))
print "add softmax layer..."
CNN_sen.add(Dense(2))

CNN_sen.add(Activation(activation='softmax'))


print "compile the model"
CNN_sen.compile(optimizer='adadelta', loss=KLDivergence)

print "training..."

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

model_path = './model/%s.hdf5' % dataset_name
best_model = ModelCheckpoint(filepath=model_path, monitor='val_loss',
    save_best_only=True, mode='min')

CNN_sen.fit(x=train_X_sen, y=train_strength,
            batch_size=batch_size,
            nb_epoch=30,
            validation_data=(val_X_sen, val_strength),
            callbacks=[early_stopping, best_model])
print "load the best model from disk..."
CNN_sen.load_weights(model_path)

pred_strength = CNN_sen.predict(x=test_X_sen,
    batch_size=batch_size)

print "evaluate performance of the system..."
accu, ap, rmse = evaluate(strength_gold=test_strength, strength_pred=pred_strength)
print "accuracy: %s, ap: %s, rmse: %s" % (accu, ap, rmse)
"""


pred_strengths_lines = []
for strength in pred_strength:
    pred_strengths_lines.append('%s\n' % ' '.join([str(ele) for ele in strength]))
with open('./res/%s_pred.txt' % dataset_name, 'w+') as fp:
    fp.writelines(pred_strengths_lines)
