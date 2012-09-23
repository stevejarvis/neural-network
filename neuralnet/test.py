'''
Created on Sep 19, 2012

@author: steve

Now do your tests first, honey.

Write temp files to end in '.test' and they'll be 
cleaned up.
'''
import unittest
import neuralnet
import os
import glob
import random
import decimal

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

    def testBadInputLoadWeights(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        goodpath = './weights_good.test'
        badpath = './weights_bad.test'
        # File should be accepted
        with open(goodpath, 'w') as f:
            print('0.32 .7', file=f)
            print('0.56 .49', file=f)
            print('0.32', file=f)
        # File should be rejected
        with open(badpath, 'w') as f:
            print('0.32 .7', file=f)
            print('0.56', file=f)
            print('0.32 .4', file=f)
        nn.load_weights(goodpath)
        #TODO implement this
        '''self.assertRaises(ValueError, 
                          nn.load_weights,
                          badpath)'''
        
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
                

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()