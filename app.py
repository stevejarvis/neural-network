'''
Created on Sep 19, 2012

@author: steve

Example app using the neuralnet package. It will learn the XOR gate.
'''

import neuralnet

nn = neuralnet.NeuralNetwork(2, 2, 1)

# Generally you want different data sets for training and testing, but
# we're very limited with XOR.
data_train = data_test = [((0, 0), 0), 
                          ((0, 1), 1),
                          ((1, 0), 1),
                          ((1, 1), 0)]

# There are a handful of optional kw args, but defaults are OK.
nn.train_network(data_train, data_test, success_goal=1)