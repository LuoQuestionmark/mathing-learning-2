import MLP
import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>ii', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

X_train, y_train = load_mnist('./', kind='train')
X_test, y_test = load_mnist('./', kind='t10k')

def hyper_training(n_hidden, l2, l1, epochs, eta, alpha, decrease_const, minibatches):
    """
    train a model with given hyperparameters,
    return the test accuracy.
    """
    nn = MLP.NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=n_hidden, l2=l2, l1=l1, epochs=epochs, eta=eta, alpha=alpha, decrease_const=decrease_const, minibatches=minibatches, shuffle=True)
    nn.fit(X_train, y_train, print_progress=True)

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

    return acc

def random_layer():
    layer = random.randint(1, 3)
    return [random.randint(10, 100) for _ in range(layer)]

def log(values):
    with open("./log", 'a') as f:
        print(values, file=f)
        print("\n", file=f)

if __name__ == '__main__':
    for _ in range(10):
        n_hidden = random_layer() 
        l2 = 0.1 * random.random()
        l1 = 0.1 * random.random()
        epochs = random.randint(10, 100)
        eta = 0.01 * random.random()
        alpha = 0.001 * random.random()
        decrease_const = 0.0001 * random.random()
        minipatches = random.randint(10, 50)

        acc = hyper_training(n_hidden, l2, l1, epochs, eta, alpha, decrease_const, minipatches)

        values = dict(
            n_hidden = n_hidden,
            l2 = l2,
            l1 = l1,
            epochs = epochs,
            eta = eta,
            alpha = alpha,
            decrease_const = decrease_const,
            minipatches = minipatches,
            acc = acc
        )
        log(values)


        

# nn = MLP.NeuralNetMLP(n_output=10,
#                   n_features=X_train.shape[1],
#                   n_hidden=[50, 50],            # changeable
#                   l2=0.1,                       # changeable
#                   l1=0.0,                       # changeable
#                   epochs=50,                    # changeable, but for the performance we will keep it little
#                   eta=0.001,                    # changeable
#                   alpha=0.001,                  # changeable
#                   decrease_const=0.00001,       # changeable
#                   minibatches=50,               # changeable
#                   shuffle=True,
#                   random_state=1)

# nn.fit(X_train, y_train, print_progress=True)


# batches = np.array_split(range(len(nn.cost_)), 1000)
# cost_ary = np.array(nn.cost_)
# cost_avgs = [np.mean(cost_ary[i]) for i in batches]


# y_train_pred = nn.predict(X_train)
# acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
# print('Training accuracy: %.2f%%' % (acc * 100))

# y_test_pred = nn.predict(X_test)
# acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
# print('Test accuracy: %.2f%%' % (acc * 100))
