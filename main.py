#import tflearn

from processor import generateSamples

samples1X, samples1Y = generateSamples(1)
samples3X, samples3Y = generateSamples(3)
samples5X, samples5Y = generateSamples(5)

#SF AB GIDP RBI G HR H BB SH HBP R SO 2B SB CS 3B IBB

print samples1X[0]

input_ = tflearn.input_data(shape=[None, len(samples1X[0])])
net = tflearn.fully_connected(input_, 32)
net = tflearn.fully_connected(net, 32)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)

m = tflearn.DNN(regression)
m.fit(samples1X, samples1Y["H"])