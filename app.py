'''
Created on Sep 19, 2012

@author: steve

Example app using the neuralnet package. It will learn the XOR gate.
'''

import neuralnet

''' Define the neural net slightly differently. Instead of interpreting a
single output as a 1 or 0, have two outputs, one representing 1, the 
second 0.'''
nn = neuralnet.NeuralNetwork(2, 3, 2)

# Generally you want different data sets for training and testing, but
# we're very limited with XOR.
data_train = [((0, 0), (0, 1)), 
              ((0, 1), (1, 0)),
              ((1, 0), (1, 0)),
              ((1, 1), (0, 1))]

for n in range(1000):
    # There are a handful of optional kw args, but defaults are OK.
    nn.train_network(data_train, iters=5, momentum=0.3, change_rate=0.7)
    
    for i in range(4):
        out = nn.evaluate(data_train[i][0])
        print('data[{}] -> {}'.format(data_train[i][0], out[0]))
        
    print('\n******************************\n')