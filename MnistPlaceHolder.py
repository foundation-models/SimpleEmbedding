from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


def load():
    """
    :return: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    return mnist.load_data()


class MnistPlaceHolder:
    def __init__(self):
        (self.x_train, _), (_, _) = load()

    @staticmethod
    def get_model(encoding_dim=256):
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        return Model(input_img, encoded)

    def plot(self):
        plt.imshow(self.x_train[0])
        plt.show()

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
    plt.show()
    myclass = MnistPlaceHolder()
    myclass.plot()
    print('------ Done ----')
