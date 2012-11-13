'''
Created on Sep 19, 2012

@author: steve

Now do your tests first, honey.

Write temp files to end in '.test' and they'll be 
cleaned up.
'''
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

    def testBadInputLoadWeights(self):
        nn = neuralnet.NeuralNetwork(2, 2, 1)
        badpath = './weights_bad.test'
        with open(badpath, 'w') as f:
            print('0.32 .7', file=f)
            print('0.56', file=f)
            print('0.32 .4', file=f)
        self.assertRaises(ValueError, 
                          nn.load_weights,
                          badpath)
        
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
        nn2 = neuralnet.NeuralNetwork(2, 3, 2)
        self.assertRaises(ValueError, 
                          nn2.load_weights,
                          './save.test')
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
