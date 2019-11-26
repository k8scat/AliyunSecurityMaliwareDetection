import pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, Embedding, Input, SpatialDropout1D
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np
import csv

# 读取持久化的对象
with open('data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    test_data = pickle.load(f)

max_len = 6000

labels = to_categorical(np.asarray(train_data['label'].tolist()), num_classes=8)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['apis'].tolist())
tokenizer.fit_on_texts(test_data['apis'].tolist())

vocab = tokenizer.word_index
x_train_word_ids = tokenizer.texts_to_sequences(train_data['apis'].tolist())
x_test_word_ids = tokenizer.texts_to_sequences(test_data['apis'].tolist())

x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=max_len)

x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=max_len)


def text_cnn():
    kernel_size = [2, 4, 6, 8, 10]
    conv_activation = 'relu'
    _input = Input(shape=(max_len,), dtype='int32')
    _embed = Embedding(304, 256, input_length=max_len)(_input)
    _embed = SpatialDropout1D(0.15)(_embed)
    warppers = []
    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=32, kernel_size=_kernel_size, activation=conv_activation, padding='same')(_embed)
        warppers.append(MaxPool1D(2)(conv1d))

    fc = concatenate(warppers)
    fc = Flatten()(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.5)(fc)
    _preds = Dense(8, activation='softmax')(fc)
    _model = Model(inputs=_input, outputs=_preds)
    _model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    return _model


def dila():
    main_input = Input(shape=(max_len,), dtype='float64')
    _embed = Embedding(304, 256, input_length=max_len)(main_input)
    _embed = SpatialDropout1D(0.25)(_embed)
    wrappers = []
    num_filters = 64
    kernel_size = [2, 3, 4, 5]
    conv_activation = 'relu'
    for _kernel_size in kernel_size:
        for dilated_rate in [1, 2, 3, 4]:
            conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation=conv_activation,
                            dilation_rate=dilated_rate)(_embed)
            wrappers.append(GlobalMaxPooling1D()(conv1d))

    fc = concatenate(wrappers)
    fc = Dropout(0.5)(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.25)(fc)
    _preds = Dense(8, activation='softmax')(fc)

    _model = Model(inputs=main_input, outputs=_preds)

    _model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    return _model


def fasttext():
    main_input = Input(shape=(max_len,), dtype='float64')
    embedder = Embedding(304, 256, input_length=max_len)
    embed = embedder(main_input)
    # cnn1模块，kernel_size = 3
    gb = GlobalAveragePooling1D()(embed)
    main_output = Dense(8, activation='softmax')(gb)
    _model = Model(inputs=main_input, outputs=main_output)
    return _model


X, y = x_train_padded_seqs, labels

# 将原训练集分成模型的训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = dila()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', baseline=None,
                    restore_best_weights=False)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=1,
                    shuffle=True,
                    validation_data=(X_val, y_val))

pred_val = model.predict(X_val)
pred_test = model.predict(x_test_padded_seqs)

preds = model.predict(x_test_padded_seqs)
out = []
for i in range(test_data.shape[0]):
    tmp = []
    probs = preds[i].tolist()
    # file_id
    tmp.append(i + 1)
    tmp.extend(probs)
    out.append(tmp)

with open('result_textcnn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
    writer.writerows(out)
