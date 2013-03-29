'''
Created on Sep 19, 2012

@author: steve

A flexible, neat neural network.
'''

from __future__ import print_function
from random import random
import math

class NeuralNetwork(object):
    '''
    The public interface is meant to be used like this:

    nn = NeuralNet(number_in, number_hid, number_out)
    while not_satisfied:
        nn.train(data, change_rate, momentum_rate, iterations)
    answer = nn.evaluate(inputs)   
    nn.save_weights('/some/path')
    nn.load_weights('/some/path')
    '''
    
    def __init__(self, nin, nhid, nout):
        # Check for valid input
        for param in [nin, nhid, nout]:
            if param == 0:
                raise ValueError('Must have more than one node for each layer')
            elif not isinstance(param, int):
                raise ValueError('Dimensions of network must be ints.')
        
        # Add one to the input layer to act as a bias. The bias helps ensure
        # that the network is able to learn the related function by shifting
        # the graph as necessary.
        self.num_input = nin + 1
        self.num_hidden = nhid
        self.num_output = nout
        
        self.weights_hid_one = self._make_matrix(self.num_input, self.num_hidden)
        self.weights_hid_two = self._make_matrix(self.num_hidden, self.num_hidden)
        self.weights_out = self._make_matrix(self.num_hidden, self.num_output)

        self.momentum_hid_one = self._make_matrix(self.num_input, self.num_hidden, 0.0)
        self.momentum_hid_two = self._make_matrix(self.num_hidden, self.num_hidden, 0.0)
        self.momentum_out = self._make_matrix(self.num_hidden, self.num_output, 0.0)
        
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
        cross = self._dot(self.activation_in, self.weights_hid_one)
        self.activation_hid_one = [self._tanh(val) for val in cross]
        
        # Second hidden layer
        cross = self._dot(self.activation_hid_one, self.weights_hid_two)
        self.activation_hid_two = [self._tanh(val) for val in cross]
        
        # Output activations just like the hidden layers'.
        cross = self._dot(self.activation_hid_two, self.weights_out)
        self.activation_out = [self._tanh(val) for val in cross]
            
        return self.activation_out
                     
    def _back_propagate(self, target, change_mult, momentum_mult):
        '''Work from the output of the network back up adjusting
        weights to inch nearer the connections (and therefore the answers) we
        want. '''
        # Target could have been passed as an int, but needs to be expandable
        if type(target) is int:
            target = [target]
        
        # First calculate deltas of the output weights. 
        # delta = (expected - actual) * d(tanh(a))/da
        delta_out = [0.0] * self.num_output
        for j in range(self.num_output):
            error = target[j] - self.activation_out[j]
            delta_out[j] = error * self._derivative_tanh(self.activation_out[j])
        
        # Update the output weights
        self._update_weights(self.weights_out, 
                             self.num_hidden, 
                             self.activation_hid_two, 
                             self.num_output, 
                             delta_out, 
                             self.momentum_out, 
                             change_mult, 
                             momentum_mult)
        
        # Find deltas for the second hidden layer.
        delta_hid_two = self._calc_deltas(self.num_hidden, 
                                          self.activation_hid_two, 
                                          self.num_output, 
                                          delta_out, 
                                          self.weights_out)
        
                
        # Update the weights for hidden layer.
        self._update_weights(self.weights_hid_two, 
                             self.num_hidden, 
                             self.activation_hid_one, 
                             self.num_hidden, 
                             delta_hid_two, 
                             self.momentum_hid_two, 
                             change_mult, 
                             momentum_mult)
         
        # After the hid two weights change, find deltas for hid1.       
        delta_hid_one = self._calc_deltas(self.num_hidden, 
                                          self.activation_hid_one, 
                                          self.num_hidden, 
                                          delta_hid_two, 
                                          self.weights_hid_two)
         
        # And update the hid one weights 
        self._update_weights(self.weights_hid_one, 
                             self.num_input, 
                             self.activation_in, 
                             self.num_hidden, 
                             delta_hid_one, 
                             self.momentum_hid_one, 
                             change_mult, 
                             momentum_mult)
              
    def _calc_deltas(self, number_nodes, activations, number_nodes_downstream, 
                     deltas_downstream, weights_downstream):
        # Calculate the deltas of the hidden layer.
        # delta = sum(downstream weights * deltas) * d(tanh(a))/da
        deltas = [0.0] * number_nodes
        for j in range(number_nodes):
            error = 0.0
            # This inner loop sums all errors downstream of the current neuron
            for k in range(number_nodes_downstream):
                error += deltas_downstream[k] * weights_downstream[j][k]
            deltas[j] = error * self._derivative_tanh(activations[j])
        return deltas
    
    def _update_weights(self, changing_weights, number_nodes_upstream, 
                        activations_upstream, number_nodes_downstream, 
                        deltas_downstream, momentums, change_co, 
                        mom_co):
        # change = cofactor * delta * current_value + momentum
        # weights += changes
        for j in range(number_nodes_upstream):
            for k in range(number_nodes_downstream):
                change = change_co * deltas_downstream[k] * activations_upstream[j]
                # This works because parameters are passed by value but are
                # references to the variable. Lists are mutable, so changes will
                # be reflected IRL.
                changing_weights[j][k] += change + (mom_co * 
                                                    momentums[j][k])
                # Momentum speeds up learning by minimizing "zig zagginess".
                momentums[j][k] = change
                
    def train_network(self, data_train, change_rate=0.4, momentum=0.1, 
                      iters=1000):
        '''Train the network with repeated evaluations and back propagations.
        Data is passed as a list of input, target pairs.'''
        
        # First train the network.
        for i in range(iters):
            # Choose a random element from the training set
            selection = math.floor(random() * len(data_train))
            data = data_train[int(selection)]
            self.evaluate(data[0])
            self._back_propagate(data[1], change_rate, momentum)
    
    def load_weights(self, source):
        '''In actual implementation it would be inefficient to train the 
        network each time. Instead, save and load weights.'''
        import shelve
        d = shelve.open(source, flag='r')
        hid_one_temp = d['weights_hid_one']
        hid_two_temp = d['weights_hid_two']
        out_temp = d['weights_out']
        if (len(self.weights_hid_one) != len(hid_one_temp) 
                or len(self.weights_out) != len(out_temp)):
            raise ValueError('Wrong dimensions  on set of weights.')
        self.weights_hid_one = hid_one_temp
        self.weights_hid_two = hid_two_temp
        self.weights_out = out_temp
        d.close()
    
    def save_weights(self, dest):
        '''Save the current weights with shelve.'''
        import shelve
        d = shelve.open(dest)
        d['weights_hid_one'] = self.weights_hid_one
        d['weights_hid_two'] = self.weights_hid_two
        d['weights_out'] = self.weights_out
        d.close()
    
    def _make_matrix(self, depth, breadth, fill=None):
        matrix = []
        if fill is None:
            for row in range(depth):
                matrix.append([])
                for col in range(breadth):
                    matrix[len(matrix) - 1].append(0.5 * (random() - 0.5))
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
