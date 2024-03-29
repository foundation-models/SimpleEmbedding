from unittest import TestCase
from MnistPlaceHolder import MnistWrapper
from MnistPlaceHolder import Regression
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import warnings


class TestMnistPlaceHolder(TestCase):

    def test_get_embeddings(self):
        instance = MnistWrapper()
        embeddings = instance.get_embeddings()
        plt.plot(embeddings[0])
        plt.show()
        self.assertTrue(embeddings.dtype, np.float32)

    def test_mnist_init(self):
        instance = MnistWrapper()
        plt.imshow(instance.x_train[0])
        plt.show()
        self.assertIsNotNone(instance.x_train, msg='could not load mnist')
        self.assertEqual(instance.x_train.shape, (60000, 28, 28))
        self.assertEqual(max(instance.x_train[0].reshape(-1)), 255)
        self.assertEqual(instance.x_train.dtype, np.uint8)

    def test_sigmoid(self):
        self.assertEqual(0.5, Regression.sigmoid(0))

    def test_propagate(self):
        w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
        instance = Regression(X,Y)
        dw, db, cost = instance.propagate(w,b)
        self.assertTrue(np.allclose([[0.99993216],[1.99980262]],dw))
        self.assertTrue(np.allclose(0.49993523062470574, db))
        self.assertTrue(np.allclose(6.000064773192205, cost))

    def test_train(self):
        w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
        instance = Regression(X,Y,w,b)
        costs = instance.train(num_iterations=10000, learning_rate=0.009)
        self.assertTrue(np.allclose([[-5.03678642],[0.8384754]],instance.W))
        self.assertTrue(np.allclose(4.437630910971542, instance.bias))
        # print(instance.W)
        # print(instance.bias)
        # print(costs[0:10])
        # print(costs[-10:])
        self.assertTrue(np.allclose(instance.predict(X),Y))

    def test_mnist_prediction(self):
        mnist = MnistWrapper()
        m = mnist.X_train.shape[0]
        model = Regression(mnist.X_train.reshape(m,-1).T/255., mnist.Y_train)
        print(model.X.shape)
        print(model.Y.shape)
        print(model.W.shape)
        costs = model.train(100, 0.009, 1)
        plt.plot(costs)
        plt.show()
        print(costs[0:10])
        print(costs[-10:])
        print(mnist.X_test.shape)
        print(mnist.Y_test.shape)
        predicted = model.predict(mnist.X_test.reshape(mnist.X_test.shape[0],-1).T/255.)
        print(predicted.shape)
        print(np.max(predicted - mnist.Y_test), np.min(predicted - mnist.Y_test))
        self.assertTrue(True)

