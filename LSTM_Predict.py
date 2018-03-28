from __future__ import print_function

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import pandas as pd

warnings.filterwarnings("ignore")


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
    print('data len:', len(data))
    print('sequence len:', seq_len)

    sequence_len = seq_len + 1
    result = []
    for index in range(len(data) - sequence_len):
        result.append(data[index: index + sequence_len])

    print('result len:', len(result))
    print('result shape:', np.array(result).shape)
    print(result[:1])

    if normalise_window:
        result = normalise_windows(result)

    print(result[:1])
    print('normalise_windows result shape:', np.array(result).shape)

    result = np.array(result)

    # 划分train, test
    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

# 模型多个中间层
def build_model(layers):
    model = Sequential()
    # 第一中间层
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    # 第二中间层
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    # 第三中间层
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))     # 激励函数
    # 时间层
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Comploication Time :", time.time() - start)
    return model


# 直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape)
    predicted = np.reshape(predicted, (predicted.size, ))
    return predicted


# 滚动预测
def predict_sequence_full(model, data, window_size):
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


# 滑动窗口+滚动预测
def predict_sequence_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_result(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig('C:/'+filename+'.png')


def plot_result_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    plt.savefig('plot_result_multiple.png')


if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 50

    print('>Loading data ..')

    x_train, y_train, x_test, y_test = load_data('./sp500.csv', seq_len, True)

    print('X train shape:', x_train.shape)
    print('y train shape:', y_train.shape)
    print('X test shape:', x_test.shape)
    print('y test shape:', y_test.shape)

    print('>Data Loaded. Compiling ...')

    model = build_model([1, 50, 100, 1])

    model.fit(x_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)
    multiplt_predictions = predict_sequence_multiple(model, x_test, seq_len, prediction_len=50)
    print('multiple_predictions shape:', np.array(multiplt_predictions).shape)

    full_predictions = predict_sequence_full(model, x_test, seq_len)
    print('full_predictions shape:', np.array(full_predictions).shape)

    point_by_point_predictions = predict_point_by_point(model, x_test)
    print('point_by_point_predictions:', np.array(point_by_point_predictions).shape)

    print('Training duration(s):', time.time() - global_start_time)

    plot_result_multiple(multiplt_predictions, y_test, 50)
    plot_result(full_predictions, y_test, 'full_predictions')
    plot_result(point_by_point_predictions, y_test, 'point_by_point_predictons')
