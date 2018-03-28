from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

# Sequential model的第一层必须包含input_shape
# 因为必须通过这个来告诉模型你输入的是什么类型的数据，而后续的中间层就不需要这个了
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
# The Merge layer
from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

final_mode = Sequential()
final_mode.add(merged)
final_mode.add(Dense(10, activation='softmax'))


# compilation
# 该步骤的主要任务是编译训练过程，在编译过程中回调用optimizer、loss、metrics等
# 其中optimizer属性的主要功能是优化函数，loss是编译过程中用的损失函数。
# Metrics属性是否要求精确，通常该属性只有accuracy这一属性值，或者没有。
# 以下三种编译方法的区别主要在于损失函数的不同，损失函数通常有categorical_crossentropy、binary_crossentropy、mse
# for a multi_class classification problem
# categorical_crossentropy:亦称作多类的对数损失，注意使用该目标函数时，需要将标签
# 转化为形如(nb_samples, nb_classes)的二值序列
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# for a binary classification problem
# binary_crossentrop 亦称作对数损失
model.compile(optimizer='rmsprop',
              loss='binary_crossentrop',
              metrics=['aaccuracy'])

# for a mean squared error regression problem
model.compile(optimizer='rmsprop', loss='mse')