import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

df = pd.read_csv('./stock_dataset.csv', engine='python')
data = np.array(df['最高价'])      # 获取最高价序列
data = data[::-1]       # 反转，使数据按照日期先后顺序排列
# 以折线图展示data
plt.figure()
plt.plot(data)
# plt.show()
normalize_data = (data - np.mean(data)) / np.std(data)      # 标准化
normalize_data = normalize_data[:, np.newaxis]      # 增加数据维度
# 形成训练集
time_step = 20      # 时间步
rnn_unit = 10       # 隐藏层
batch_size = 60     # 每一批训练多少样例
input_size = 1      # 输入层
output_size = 1     # 输出层
lr = 0.0006         # 学习率

train_x, train_y = [], []

for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i: i + time_step]
    y = normalize_data[i + 1: i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# 定义神经网络
x = tf.placeholder(tf.float32, [None, time_step, input_size])       # 每批次输入网络的tensor
y = tf.placeholder(tf.float32, [None, time_step, output_size])      # ，诶批次tensor对应的标签
# 输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

# 定义LSTM网络
def lstm(batch):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(x, [-1, input_size])     # 需要将Tensor转化为2维计算，计算结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])        # 将tensor转化为3维数据，作为rnn cell 的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # output_rnn是记录LSTM每一个输出节点的结果，final_state是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])     # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states
# 训练模型
def train_lstm():
    global batch_size
    pred, _ = lstm(batch_size)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练10000次
        for i in range(100):
            step = 0
            start = 0
            end = start + batch_size
            while(end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={x: train_x[start: end], y: train_y[start: end]})
                start += batch_size
                end = start + batch_size
                # 每十步保存一次参数
                if step% 10 == 0:
                    print(i, step, loss_)
                    print("model_save:", saver.save(sess, 'model_save\\model.ckpt'))
                step += 1
train_lstm()
# 预测模型
def prediction():
    pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save\\model.ckpt')
        prev_seq = train_x[-1]
        predict = []
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={x: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, coloe='r')
        plt.show()

prediction()
