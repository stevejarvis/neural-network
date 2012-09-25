'''
Created on Sep 19, 2012

@author: steve

A flexible, neat neural network.

The public interface is meant to be used like this:

nn = NeuralNet(number_in, number_hid, number_out)
while not_satisfied:
    nn.train(data, change_rate, momentum_rate, iterations)
answer = nn.evaluate(inputs)   
nn.save_weights('/some/path')
nn.load_weights('/some/path')
'''

from random import random
import math

class NeuralNetwork(object):
    '''
    This is, uh, the important part.
    '''
    
    def __init__(self, nin, nhid, nout):
        # Check for valid input
        if 0 in [nin, nhid, nout]:
            raise ValueError('Must have more than one node for each layer')
        
        # Add one to the input layer to act as a bias. The bias helps ensure
        # that the network is able to learn the related function by shifting
        # the graph as necessary.
        self.num_input = nin + 1
        self.num_hidden = nhid
        self.num_output = nout
        
        self.weights_hid = self._make_matrix(self.num_input, self.num_hidden)
        self.weights_out = self._make_matrix(self.num_hidden, self.num_output)

        self.momentum_hid = self._make_matrix(self.num_input, self.num_hidden, 
                                              0.0)
        self.momentum_out = self._make_matrix(self.num_hidden, self.num_output, 
                                              0.0)
        
    def evaluate(self, input_vals):
        '''Will return a list of guesses per neuron based on the information 
        we have.'''
        # Can be passed as a tuple.
        input_vals = list(input_vals)
        if len(input_vals) != self.num_input - 1:
            raise ValueError('''Input values don\'t mesh with the number of 
                            neurons.''')
        # The bias neuron
        input_vals.append(1)
        
        self.activation_in = input_vals
        
        # Find the hidden layers activation levels. The activation
        # levels are the sum of products of weights and neurons in (the dot
        # product). Then tanh yields the actual activation value.
        cross = self._dot(self.activation_in, self.weights_hid)
        self.activation_hid = [self._tanh(val) for val in cross]
        
        # Find the output activations just like the hidden layer's.
        cross = self._dot(self.activation_hid, self.weights_out)
        self.activation_out = [self._tanh(val) for val in cross]
            
        return self.activation_out
                     
    def _back_propagate(self, target, change_mult, momentum_mult):
        '''Work from the output of the network back up adjusting
        weights to inch nearer the connections (and therefore the answers) we
        want.'''
        # Target could have been passed as an int, but needs to be expandable
        if type(target) is int:
            target = [target]
        
        # First calculate deltas of the output weights. 
		# delta = (expected - actual) * d(tanh(a))/da
        delta_out = [0.0] * self.num_output
        for j in range(self.num_output):
            error = target[j] - self.activation_out[j]
            delta_out[j] = error * self._derivative_tanh(self.activation_out[j])
        
        # Calculate the deltas of the hidden layer.
		# delta = sum(downstream weights * deltas) * d(tanh(a))/da
		#
		# Slightly more complicated than output because of the need to consider
		# all connected neurons further down the chain. Each neurons expected
		# output is a minimization of the collective downstream errors.
        delta_hid = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            error = 0.0
			# This inner loop sums all errors downstream of the current neuron
            for k in range(self.num_output):
                error += delta_out[k] * self.weights_out[j][k]
            delta_hid[j] = error * self._derivative_tanh(self.activation_hid[j])
                
        # Then adjust the weights of the output.
		#
		# change = cofactor * delta * current_value + momentum
		# weights += changes
        for j in range(self.num_hidden):
            for k in range(self.num_output):
                change = change_mult * delta_out[k] * self.activation_hid[j]
                self.weights_out[j][k] += change + (momentum_mult * 
                                                    self.momentum_out[j][k])
                # Momentum speeds up learning by minimizing "zig zagginess".
                self.momentum_out[j][k] = change
        
        # Update the weights for hidden layer in the same way as the output.
        for j in range(self.num_input):
            for k in range(self.num_hidden):
                change = change_mult * delta_hid[k] * self.activation_in[j]
                self.weights_hid[j][k] += change + (momentum_mult * 
                                                    self.momentum_hid[j][k])
                self.momentum_hid[j][k] = change
                
    def train_network(self, data_train, change_rate=0.4, momentum=0.1, 
                      iters=1000):
        '''Train the network with repeated evaluations and back propagations.
        Data is passed as a list of input, target pairs.'''
        
        # First train the network.
        for i in range(iters):
            # Choose a random element from the training set
            selection = math.floor(random() * len(data_train))
            data = data_train[selection]
            self.evaluate(data[0])
            self._back_propagate(data[1], change_rate, momentum)
    
    def load_weights(self, source):
        # TODO
        pass
    
    def save_weights(self, dest):
        # TODO
        pass
    
    def _make_matrix(self, depth, breadth, fill=None):
        matrix = []
        if fill is None:
            for row in range(depth):
                matrix.append([])
                for col in range(breadth):
                    matrix[len(matrix) - 1].append(random() - 0.5)
        else:
            for row in range(depth):
                matrix.append([])
                for col in range(breadth):
                    matrix[len(matrix) - 1].append(fill)
        return matrix
    
    def _tanh(self, x):
        '''Return the hyberbolic tangent of x. Tanh produces a nice sigmoidal
        function to use for the evaluations.'''
        return ((math.e ** (2 * x)) - 1) / ((math.e ** (2 * x)) + 1)
    
    def _derivative_tanh(self, y):
        '''Given the activation value of a neuron (the output of tanh(x))
        return the derivative of tanh for that y.'''
        # Proof this is equal to the derivative of tanh(x):
        # let y = tanh(x)
        # d/dx tanh(x) = sech^2(x)    // From Wolfram
        # sech^2(x) = 1/cosh^2(x)
        # sech^2(x) = (cosh^2(x) - sinh^2(x)) / cosh^2(x)
        # sech^2(x) = 1 - sinh^2(x) / cosh^2(x)
        # sech^2(x) = 1 - tanh^2(x)
        # sech^2(x) = 1 - y^2        // Substitute given. Boom.
        return 1 - (y ** 2)
    
    def _dot(self, m1, m2):
        '''Specific dot function. m1 must be the activation list, m2
        must be a matrix with depth equal to len(m1)'''
        if len(m1) != len(m2):
            raise ValueError('Can\'t dot those matrices, size matters.')
        
        new_matrix = []
        for j in range(len(m2[0])):
            dot = 0.0
            for k in range(len(m1)):
                dot += m1[k] * m2[k][j]
            new_matrix.append(dot)
        return new_matrix
