#################################
# Your name: Ido Friedman
#################################


import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    d = data.shape[1]
    w = np.zeros(d)

    for t in range(1, T + 1):
        eta_t = eta_0 / t

        i = np.random.randint(0, len(labels))

        x_i = data[i]
        y_i = labels[i]

        if y_i * np.dot(w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w
    return w


#################################

def accuracy(predictions, labels):
    return np.mean(predictions == labels)


def cross_validate_eta(train_data, train_labels, val_data, val_labels, C, T):
    # eta_values = [10 ** i for i in range(-5, 6)]  # eta_0 = 10^-5 to 10^5
    eta_values = np.linspace(0.1, 20, 100)  # 100 evenly spaced points between 0.1 and 20

    accuracies = []

    for eta_0 in eta_values:
        avg_acc = 0
        for _ in range(20):  # Average over 10 runs
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            val_predictions = np.sign(val_data @ w)
            avg_acc += accuracy(val_predictions, val_labels)
        accuracies.append(avg_acc / 20)

    plt.figure(figsize=(10, 6))
    plt.plot(eta_values, accuracies, marker='o')
    plt.xlabel("eta_0")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Initial Learning Rate (eta_0)")
    plt.grid()
    plt.show()

    best_eta_index = np.argmax(accuracies)
    best_eta_0 = eta_values[best_eta_index]
    print(f"Best eta_0: {best_eta_0}, Best Validation Accuracy: {accuracies[best_eta_index]:.4f}")
    return best_eta_0

    # # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(-5, 6), accuracies, marker='o')
    # plt.xticks(range(-5, 6), [f"10^{i}" for i in range(-5, 6)])
    # plt.xlabel("log10(eta_0)")
    # plt.ylabel("Validation Accuracy")
    # plt.title("Validation Accuracy vs Initial Learning Rate (eta_0)")
    # plt.grid()
    # plt.show()


def cross_validation_C(train_data, train_labels, val_data, val_labels, eta_0, T):
    C_values = [10 ** i for i in range(-5, 6)]  # Test C from 10^-5 to 10^5
    accuracies = []

    for C in C_values:
        avg_acc = 0
        for _ in range(20):  # Average over 10 runs
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            val_predictions = np.sign(val_data @ w)
            avg_acc += accuracy(val_predictions, val_labels)
        accuracies.append(avg_acc / 20)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, accuracies, marker='o')
    plt.xlabel("C (log scale)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Regularization Parameter (C)")
    plt.grid()
    plt.show()

    best_C_index = np.argmax(accuracies)
    best_C = C_values[best_C_index]
    print(f"Best C: {best_C}, Best Validation Accuracy: {accuracies[best_C_index]:.4f}")
    return best_C


train_data, train_labels, val_data, val_labels, test_data, test_labels = helper()
T = 1000
C = 1

eta_0 = cross_validate_eta(train_data, train_labels, val_data, val_labels, C, T)

T = 10
c = cross_validation_C(train_data, train_labels, val_data, val_labels, eta_0, T)

best_C, best_eta_0 = c, eta_0

T = 20000
w = SGD_hinge(train_data, train_labels, best_C, best_eta_0, T)

image = w.reshape((28, 28))

plt.imshow(image, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.title("Visualization of Weights (w)")
plt.show()

test_predictions = np.sign(test_data @ w)
acc = accuracy(test_predictions, test_labels)
print("accuracy of the best classifier on the test set: " + str(acc))
#################################
