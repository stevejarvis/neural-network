'''
Created on Sep 19, 2012

@author: steve

Now do your tests first, honey.

Write temp files to end in '.test' and they'll be 
cleaned up.
'''

from __future__ import print_function
import sys
import unittest
import os
import glob
import random
import decimal
sys.path.append('..')
import neuralnet

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        for f in glob.glob('*.test'):
            os.remove(f)

    def testEvaluation(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        data = [((0, 0), 0),
                ((0, 1), 1),
                ((1, 0), 1),
                ((1, 1), 0)]
        for val, out in data:
            ans = nn.evaluate(val)
            for val in ans:
                assert -1 < val < 1
    
    def testBadInputInit(self):
        self.assertRaises(ValueError, 
                          neuralnet.NeuralNetwork,
                          1, 0, 1)
        
    def testMatrixMaker(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        matrix = nn._make_matrix(2, 3)
        assert len(matrix) == 2
        for row in matrix:
            assert len(row) == 3
        matrix = nn._make_matrix(3, 4, 7)
        assert matrix == [[7,7,7,7],
                          [7,7,7,7],
                          [7,7,7,7]]
        matrix = nn._make_matrix(3, 4)
        assert matrix != [[7,7,7,7],
                          [7,7,7,7],
                          [7,7,7,7]]
        for row in matrix:
            for item in row:
                assert -1 < item < 1

    def testTanh(self):
        nn = neuralnet.NeuralNetwork(3,2,1)
        assert nn._tanh(3) > 0.99
        assert nn._tanh(4) > 0.999
        assert nn._tanh(20) > 0.999
        assert nn._tanh(-3) < 0.99
        assert nn._tanh(-12) < 0.99
        assert nn._tanh(0)  == 0
        assert 0.46211 < nn._tanh(0.5) < 0.46212
        assert round(nn._tanh(0.1), 6) == round((-1 * nn._tanh(-0.1)), 6)
        
    def testEvaluationError(self):
        nn = neuralnet.NeuralNetwork(4,5,2)
        values = [1,2,3,4,5]
        self.assertRaises(ValueError,
                          nn.evaluate,
                          values)

    def testTraining(self):
        # This test really just tests for crashes
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        data = [((0, 0), 0),
                ((0, 1), 1),
                ((1, 0), 1),
                ((1, 1), 0)]
        nn.train_network(data)
        
    def testOverflow(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        for i in range(10):
            val = decimal.Decimal(random.random())
            tan = nn._tanh(float(val))
            assert -1 < tan < 1
            
    def testDot(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        m1 = [1, 2, 3]
        m2 = [(1,2),
              (3,4),
              (4,3)]
        assert nn._dot(m1, m2) == [19, 19]
        m1 = [1,2,3,4]
        self.assertRaises(ValueError,
                          nn._dot,
                          m1, m2)
        
    def testOverall(self):
        nn = neuralnet.NeuralNetwork(2, 3, 2)
        data = [((0, 0), (0, 1)), 
              ((0, 1), (1, 0)),
              ((1, 0), (1, 0)),
              ((1, 1), (0, 1))]
        for n in range(10):
            nn.train_network(data, iters=1000, change_rate=0.5, momentum=0.2)
        out = nn.evaluate(data[0][0])
        assert out[0] < 0.2 and out[1] > 0.8
        out = nn.evaluate(data[1][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn.evaluate(data[2][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn.evaluate(data[3][0])
        assert out[0] < 0.2 and out[1] > 0.8
        
    def testSaveLoadWeightsFunctionality(self):
        nn = neuralnet.NeuralNetwork(2, 3, 2)
        data = [((0, 0), (0, 1)), 
              ((0, 1), (1, 0)),
              ((1, 0), (1, 0)),
              ((1, 1), (0, 1))]
        for n in range(10):
            nn.train_network(data, iters=1000, change_rate=0.5, momentum=0.2)
        out = nn.evaluate(data[0][0])
        assert out[0] < 0.2 and out[1] > 0.8
        out = nn.evaluate(data[1][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn.evaluate(data[2][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn.evaluate(data[3][0])
        assert out[0] < 0.2 and out[1] > 0.8
        nn.save_weights('./weights.test')
        nn2 = neuralnet.NeuralNetwork(2, 3, 2)
        nn2.load_weights('./weights.test')
        out = nn2.evaluate(data[0][0])
        assert out[0] < 0.2 and out[1] > 0.8
        out = nn2.evaluate(data[1][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn2.evaluate(data[2][0])
        assert out[0] > 0.8 and out[1] < 0.2
        out = nn2.evaluate(data[3][0])
        assert out[0] < 0.2 and out[1] > 0.8
       
    def testSaveLoad(self):
        nn = neuralnet.NeuralNetwork(2, 3, 2)
        nn.save_weights('./save.test')
        nn2 = neuralnet.NeuralNetwork(2, 3, 2)
        nn2.load_weights('./save.test')
        assert nn.weights_hid == nn2.weights_hid
        assert nn.weights_out == nn2.weights_out
        
    def testBadWeights(self):
        nn = neuralnet.NeuralNetwork(2, 4, 2)
        nn.save_weights('./save.test')
        nn2 = neuralnet.NeuralNetwork(1, 7, 1)
        self.assertRaises(ValueError, 
                          nn2.load_weights,
                          './save.test')
        
    def testHugeNetwork(self):
        ''' Make a list of relatively simple but big data, make sure the
        nnet can learn it. '''
        import math
        import datetime
        min_size = 10
        max_size = 70
        failures = []
        times = []
        
        for size in range(min_size, max_size):
            print('On %d...' %size)
            num_out = int(math.sqrt(size))
            nn = neuralnet.NeuralNetwork(size, size, num_out)
            data = []
            for i in range(max_size):
                # Consider the input the index of number 1. i.e. [0,0,0,1] is 
                # thought of as 3.
                input_bits = [1  if j == i else 0 for j in range(size)]
                # Learn math: answer = root(input_index) - input_index
                # In other words: answer[input_index^2 + input_index] = 1
                answer = [1 if input_bits[j*j+j] == 1 else -1 for j in range(num_out)]
                data.append((input_bits, answer))
                
            start_time = datetime.datetime.now()
            for n in range(5):
                nn.train_network(data, 
                                 iters=1000, 
                                 change_rate=0.5, 
                                 momentum=0.2) 
            t = (datetime.datetime.now() - start_time).total_seconds() 
            times.append(t)      
            
            # Var to count number of failures for each size
            num_failures = 0
            for set in data:
                res = nn.evaluate(set[0])
                res = [round(x) for x in res]
                #self.assertEqual(res, set[1], 'Fail with size %d' %size)
                if res != set[1]:
                    # The network got it wrong
                    num_failures += 1
            failures.append(num_failures)
            
        # Plot it so I can look at something nice in a couple hours.
        try:
            import matplotlib.pyplot as lab
        except:
            ''' Install matplotlib. '''
            pass
        else:
            x_data = range(min_size, max_size)
            lab.title('Errors & Time vs Network Size -- Constant Iterations')
            lab.xlabel('Network Size (# of input and hidden neurons)')
            lab.ylabel('Errors (on data of size %d) and Time (seconds)' %max_size)
            lab.plot(x_data, failures, color='r', label='Failures')
            lab.plot(x_data, times, color='b', label='Time (seconds)')
            lab.legend()
            lab.show()
                

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
