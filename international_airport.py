import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = read_csv('./international-airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
print(dataset.shape[0], dataset.shape[1])
dataset = dataset.astype('float32')

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
np.random.seed(7)
# 归一化数据,训练集占数据集的67%，测试集占数据集的33%
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(len(dataset))
train_size = int((len(dataset)) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# 划分测试集训练集数据,X=t, Y=t+1时的数据，并且此时的维度为[sample, feature]
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 投入到LSTM的X需要有这样的结构：[samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 创建LSTM模型
# 输入层1个input, 隐藏层有4个神经元，输出层就是一个预测值，
#激活函数使用sigmoid,迭代100次，batch_size为1

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 计算误差之前需要先把预测数据转换成同一单位
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 计算mean_squared_error
trainsocre = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.f RMSE' % (trainsocre))
testsocre = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testsocre))

# 画出结果：蓝色为原数据，绿色为训练集的预测值，红色为测试集预测数据
# 画出训练集预测数据
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# 画出测试集预测数据
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1: len(dataset) - 1, :] = testPredict
# 原始数据
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



