'''
Created on Sep 19, 2012

@author: steve

Example app using the neuralnet package. It will learn the XOR gate.
'''

import neuralnet

nn = neuralnet.NeuralNetwork(2, 2, 1)

# Generally you want different data sets for training and testing, but
# we're very limited with XOR.
data_train = [((0, 0), 0), 
              ((0, 1), 1),
              ((1, 0), 1),
              ((1, 1), 0)]

for n in range(100):
    # There are a handful of optional kw args, but defaults are OK.
    nn.train_network(data_train, iters=5, momentum=0.1, change_rate=0.5)
    
    for i in range(4):
        out = nn.evaluate(data_train[i][0])
        print('data[{}] -> {}'.format(data_train[i][0], out[0]))
        
    print('\n******************************\n')