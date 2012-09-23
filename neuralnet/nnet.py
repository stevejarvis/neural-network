'''
Created on Sep 19, 2012

@author: steve

A flexible, neat neural network.
'''

from random import random

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
        
        self.weights_hid = self._make_matrix(self.num_hidden, self.num_input)
        self.weights_out = self._make_matrix(self.num_output, self.num_hidden)
        
        self.momentum_hid = self._make_matrix(self.num_hidden, self.num_input, 
                                              0.0)
        self.momentum_out = self._make_matrix(self.num_output, self.num_hidden, 
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
        
        # Fill a list of the hidden layers activation levels. The activation
        # levels are the sum of products of weights and neurons in. Then
        # tanh yields the actual activation value.
        self.activation_hid = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            val = 0.0
            for k in range(self.num_input):
                val += self.weights_hid[j][k] * self.activation_in[k]
            self.activation_hid[j] = self._tanh(val)
        
        # Find the output activations just like the hidden layer's
        self.activation_out = [0.0] * self.num_output
        for j in range(self.num_output):
            val = 0.0
            for k in range(self.num_hidden):
                val += self.weights_out[j][k] * self.activation_hid[k]
            self.activation_out[j] = self._tanh(val)
            
        return self.activation_out
        
    def train_network(self, data_train, data_validation, change_mult=0.7, 
                      momentum_mult=0.0, success_goal=0.95, max_iter=100000):
        '''Train the network with repeated evaluations and back propagations.
        Training will continue until either the goal success rate is hit or the
        max iterations.
        
        Data is passed as a list of input, target pairs.'''
        import math
        epoch_size = max(math.ceil(max_iter / 1000), 10)
        current_rate = 0.0  # Running success rate
        iterations = 0  # Number of trainings done
        while current_rate < success_goal and iterations < max_iter:
            
            # First train the network.
            for i in range(epoch_size):
                selection = math.floor(random() * len(data_train))
                data = data_train[selection]
                self.evaluate(data[0])
                self._back_propagate(data[1], change_mult, momentum_mult)
                iterations += 1
            
            # And after training, see if it's good enough to stop.
            error_count = 0
            for values, targets in data_validation:
                output = self.evaluate(values)
                if type(targets) is int:
                    targets = [targets]
                if output != targets:
                    error_count += 1
            
            current_rate = error_count / len(data_validation)
            print('Iterations: {} Success rate: {}'.format(iterations, 
                                                           current_rate))
                
    def _back_propagate(self, target, change_mult, momentum_mult):
        '''Work from the output of the network back up adjusting outputs and
        weights to inch nearer the connections (and therefore the answers) we
        want.'''
        # Target could have been passed as an int, but needs to be expandable
        if type(target) is int:
            target = [target]
        # First calculate error and update the output weights
        error = delta_out = [0.0] * self.num_output
        for j in range(self.num_output):
            error = target[j] - self.activation_out[j]
            delta_out[j] = error * self._tanh(self.activation_out[j])
            for k in range(self.num_hidden):
                change = (change_mult * delta_out[j]) + (momentum_mult * 
                                                  self.momentum_out[j][k])
                self.weights_out[j][k] += change
                self.momentum_out[j][k] = change
        
        # Hidden is a two part process because of its sandwiched nature. 
        # First calculate the deltas.
        delta_hid = [0.0] * self.num_hidden
        for j in range(self.num_hidden):
            error = 0.0
            for k in range(self.num_output):
                error += self.weights_out[k][j] * delta_out[k]
            delta_hid[j] = error * self._tanh(self.activation_hid[j])
        
        # Update the weights for hidden layer.
        for j in range(self.num_hidden):
            for k in range(self.num_input):
                change = (change_mult * delta_hid[j]) + (momentum_mult * 
                                                        self.momentum_hid[j][k])
                self.weights_hid[j][k] += change
                self.momentum_hid[j][k] = change 
    
    def load_weights(self, source):
        # TODO
        pass
    
    def save_weights(self, dest):
        # TODO
        pass
    
    def _make_matrix(self, depth, breadth, fill=None):
        row = []
        if fill is None:
            for x in range(breadth):
                row.append(random() - 0.5)
        else:
            try:
                fill = float(fill)
            except ValueError:
                raise ValueError('Fill must be a number, duh.')
            for x in range(breadth):
                row.append(fill)
        return [row]*depth
    
    def _tanh(self, x):
        '''Return the hyberbolic tangent of x. Tanh produces a nice sigmoidal
        function to use for the evaluations.'''
        import math
        tan = (math.e ** (2 * x) - 1) / (math.e ** (2 * x) + 1)
        return tan
    
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