

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from numpy import ndarray


class MnistPlaceHolder():
    def __init__(self):
        (self.x_train, _), (_, _) = self.load()

    def load(self):
        """
        :param url:
        :return: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
        """
        return mnist.load_data()

    def getModel(self, encoding_dim=256):
        input_img = Input(shape=(28,28,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        return Model(input_img, encoded)

    def getEmbeddings(self):
        x_train_float32 = self.x_train[0:2].reshape(-1,28,28).astype(np.float32)
        model = self.getModel()
        embeddings = model.predict(x_train_float32)
        print(embeddings.shape)
        return embeddings

