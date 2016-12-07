from processor import generateSamples
import tflearn
import numpy as np

def MSE(preds, actuals):
    total = 0
    count = 0
    for i, pred in enumerate(preds):
        for j, value in enumerate(pred[0]):
            total += (pred[0][j] - actuals[i][j])**2
        count += 1

    print(count)
    return total / count

samples1X, samples1Y = generateSamples(1)
samples3X, samples3Y = generateSamples(3)
samples5X, samples5Y = generateSamples(5)

samples1XTrain = samples1X[:50000]
samples1XTest = samples1X[50000:]
samples1YTrain = samples1Y[:50000]
samples1YTest = samples1Y[50000:]

samples3XTrain = samples3X[:50000]
samples3XTest = samples3X[50000:]
samples3YTrain = samples3Y[:50000]
samples3YTest = samples3Y[50000:]

samples5XTrain = samples5X[:50000]
samples5XTest = samples5X[50000:]
samples5YTrain = samples5Y[:50000]
samples5YTest = samples5Y[50000:]

'''
"age", "year", "games", "atbats", "runs", "hits", "doubles", "triples", "homeruns", "runsbattedin", "stolenbases", "caughtstealing", "walks", "strikeouts", "intentionalwalks", "hitbypitch", "sacrificehits", "sacrificeflies", "groundintodoubleplay"
'''

MT1 = [24, 2015, 159, 575, 104, 172, 32, 6, 41, 90, 11, 7, 92, 158, 14, 10, 0, 5, 11]
MT3 = [24, 2015, 159, 575, 104, 172, 32, 6, 41, 90, 11, 7, 92, 158, 14, 10, 0, 5, 11, 23, 2014, 157, 602, 115, 173, 39, 9, 36, 111, 16, 2, 83, 184, 6, 10, 0, 10, 6, 22, 2013, 157, 589, 109, 190, 39, 9, 27, 97, 33, 7, 110, 136, 10, 9, 0, 8, 8]
MT5 = [[24, 2015, 159, 575, 104, 172, 32, 6, 41, 90, 11, 7, 92, 158, 14, 10, 0, 5, 11, 23, 2014, 157, 602, 115, 173, 39, 9, 36, 111, 16, 2, 83, 184, 6, 10, 0, 10, 6, 22, 2013, 157, 589, 109, 190, 39, 9, 27, 97, 33, 7, 110, 136, 10, 9, 0, 8, 8, 21, 2012, 139, 559, 129, 182, 27, 8, 30, 83, 49, 5, 67, 139, 4, 6, 0, 7, 7, 20, 2011, 40, 123, 20, 27, 6, 0, 5, 16, 4, 0, 9, 30, 0, 2, 0, 1, 2]]


input_ = tflearn.input_data(shape=[None, 19])
linear = tflearn.fully_connected(input_, 19)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)

m = tflearn.DNN(regression)
m.fit(samples1XTrain, samples1YTrain, n_epoch=500, show_metric=True, snapshot_epoch=False)
pred = [m.predict([x]) for x in samples1XTest]
print(MSE(pred, samples1YTest))


'''
input_ = tflearn.input_data(shape=[None, 57])
linear = tflearn.fully_connected(input_, 19)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)

m = tflearn.DNN(regression)
m.fit(samples5XTrain, samples5YTrain, n_epoch=500, show_metric=True, snapshot_epoch=False)
pred = [m.predict([x]) for x in samples3XTest]
print(MSE(pred), samples3YTest)
'''

'''
input_ = tflearn.input_data(shape=[None, 95])
linear = tflearn.fully_connected(input_, 19)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)

m = tflearn.DNN(regression)
m.fit(samples5XTrain, samples5YTrain, n_epoch=500, show_metric=True, snapshot_epoch=False)
pred = [m.predict([x]) for x in samples5XTest]
print(MSE(pred), samples5YTest)
'''
