'''
Created on Sep 19, 2012

@author: steve

'''

class NeuralNetwork(object):
    '''
    This is the meat of the matter.
    '''
    
    def __init__(self, nin, nhid, nout, bias=True):
        # Check for valid input
        if 0 in [nin, nhid, nout]:
            raise ValueError('Must have more than one node for each layer')
        
        if bias:
            # Bias will add an extra node to the input and hidden layers.
            # The addition of this node will allow the range of the network to
            # be shifted.
            nin += 1
            nhid += 1
        
        self.number_input = nin
        self.number_hidden = nhid
        self.number_output = nout
        
    def evaluate(self, input):
        '''Will return a best guess based on the information we have.'''
        pass
    
    def load_weights(self, source):
        '''Load the weights from a file if training has already been 
        done/saved. Pass path to file.'''
        with open(source, 'r') as f:
            lines = f.readlines()
        if len(lines) != 3: 
            raise ValueError('Need exactly weights for three layers.')  
    
    def save_weights(self, dest):
        pass
    
    def _make_matrix(self, depth, breadth):
        from random import random
        row = []
        for x in range(breadth):
            row.append(random())
        return [row]*depth
    
    def _tanh(self, x):
        '''Return the hyberbolic tangent of x. Tanh produces a nice sigmoidal
        function to use for the evaluations.'''
        import math
        return (math.e ** (2 * x) - 1) / (math.e ** (2 * x) + 1)
    
    def _derivative_tanh(self, x):
        '''d/dx(tanh(x)) is sech^2x is 2/(e^x + e^-x)'''
        import math
        return 2 / ((math.e ** x) + (math.e ** -x))