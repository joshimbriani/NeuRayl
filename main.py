from processor import generateSamples
import tflearn

samples1X, samples1Y = generateSamples(1)
samples3X, samples3Y = generateSamples(3)
samples5X, samples5Y = generateSamples(5)

'''
"age", "year", "games", "atbats", "runs", "hits", "doubles", "triples", "homeruns", "runsbattedin", "stolenbases", "caughtstealing", "walks", "strikeouts", "intentionalwalks", "hitbypitch", "sacrificehits", "sacrificeflies", "groundintodoubleplay"
'''

input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)

m = tflearn.DNN(regression)
m.fit(samples1X, samples1Y["hits"], n_epoch=1000, show_metric=True, snapshot_epoch=False)
