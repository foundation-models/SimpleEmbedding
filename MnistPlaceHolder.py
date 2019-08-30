from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np


class Regression(object):
    def __init__(self, X, Y):
        self.X = X #/ 255.
        self.Y = Y
        self.w = np.zeros(shape=[X.shape[1],1])
        self.b = 0


    @staticmethod
    def sigmoid(z):
        sig = 1/(1+np.exp(-z))
        return sig

    def propagate(self, w, b):
        self.w = w
        self.b = b
        m = self.X.shape[1]
        z = np.dot(self.w.T, self.X) + self.b
        A = self.sigmoid(z)
        cost = (-1 / m) * np.sum(self.Y * np.log(A) + (1 - self.Y) * np.log(1 - A))
        cost = np.squeeze(cost)
        db = 1 / m * np.sum(A - self.Y)
        dw = 1 / m * np.dot(self.X, (A - self.Y).T)
        grads = {"dw":dw, "db":db }
        return grads, cost


class MnistWrapper:
    def __init__(self):
        (self.x_train, _), (_, _) = self.load()

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