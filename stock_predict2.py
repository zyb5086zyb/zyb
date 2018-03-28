import pandas as pd
import numpy as np
import tensorflow as tf

rnn_unit = 10
input_size = 7
output_size = 1
lr = 0.0006
# 导入数据
df = pd.read_csv('./dataset_2.csv', engine='python')
data = df.iloc[:, 2:10].values
# 生成训练集和测试集
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    batch_index= []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0))\
                            /np.std(data_train, axis=0)
    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step, :7]
        y = normalized_train_data[i: i + time_step, 7, np.newaxis ]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

def get_test_data(time_step=20, test_begin=5800):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std
    size = (len(normalized_test_data) + time_step - 1) / time_step
    test_x, test_y = [], []
    for i in range(size-1):
        x = normalized_test_data[i * time_step: (i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step: (i + 1) * time_step, :7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    return mean, std, test_x, test_y
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
# 创建神经网络
def lstm(x):
    batch_size = tf.shape(x)[0]
    time_step = tf.shape(x)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(x, [-1, input_size])  # 需要将Tensor转化为2维计算，计算结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转化为3维数据，作为rnn cell 的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # output_rnn是记录LSTM每一个输出节点的结果，final_state是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states
def train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=5800):
    x = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
    y = tf.placeholder(tf.float32, [None, time_step, output_size])  # ，诶批次tensor对应的标签
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(x)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    model_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        saver.restore(sess, model_file)
        for i in range(2000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={x: train_x[batch_index[step]:batch_index[step+1]], y: train_y[batch_index[step]:batch_index[step+1]]})
            print(i, loss_)
            if i % 200 == 0:
                print("Saver Model: ", saver.save(sess, 'stock.model\\stock2.ckpt'))
train_lstm()

# 预测模型
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)-1):
          prob = sess.run(pred, feed_dict={X: [test_x[step]]})
          predict = prob.reshape((-1))
          test_predict.extend(predict)
        test_y = np.array(test_y)*std[7]+mean[7]
        test_predict = np.array(test_predict)*std[7]+mean[7]
        acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) #acc为测试集偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()
prediction()