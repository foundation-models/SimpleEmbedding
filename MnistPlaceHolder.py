from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np


class Regression(object):
    def __init__(self, X, Y, w=None, b=0):
        self.X = X #/ 255.
        self.Y = Y
        self.W = w
        if(w is None):
            self.W = np.zeros(shape=[self.X.shape[0],1])
        self.bias = b


    @staticmethod
    def sigmoid(z):
        sig = 1/(1+np.exp(-z))
        return sig

    def propagate(self, w, b):
        m = self.X.shape[1]
        z = np.dot(w.T, self.X) + b
        A = self.sigmoid(z)
        cost = (-1 / m) * np.sum(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))
        cost = np.squeeze(cost)
        db = 1 / m * np.sum(A - self.Y)
        dw = 1 / m * np.dot(self.X, (A - self.Y).T)
        return dw, db, cost

    def train(self, num_iterations, learning_rate):
        costs = []
        w = self.W
        b = self.bias
        for i in range(num_iterations):
            dw, db, cost = self.propagate(w, b)
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if(i%100 == 0):
                costs.append(cost)
                #print('cost after {} iteration is {}'.format(i, cost))
        self.W = w
        self.bias = b
        return costs

    def predict(self, new_X):
        z = np.dot(self.W.T,new_X) + self.bias
        A = self.sigmoid(z)
        prediction = np.zeros(shape=[1,new_X.shape[1]])

        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                prediction[0, i] = 1
            else:
                prediction[0, i] = 0
        return prediction


class MnistWrapper:
    def __init__(self):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = self.load()

    @staticmethod
    def load():
        """
        :return: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
        """
        return mnist.load_data()

    @staticmethod
    def get_model(encoding_dim=256):
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        return Model(input_img, encoded)

    def get_embeddings(self):
        x_train_float32 = self.x_train[0:2].reshape(2, -1).astype(np.float32)
        print('x_train flatten shape ', x_train_float32.shape)
        print('x[0,0:20] ', x_train_float32[0,0:20])
        model = self.get_model()
        embeddings = model.predict(x_train_float32)
        print('embed..[0,0:20] ', embeddings[0,0:20])
        print("embedding shape ", embeddings.shape)
        return embeddings


if __name__ == "__main__":
    print('Done')