import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


C = 10
n = 100

# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1, X2], axis=1)
y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])

# a
#######################
# Linear kernel
linear_svm = svm.SVC(C=10, kernel="linear")

# Polynomial kernel (degree 2)
poly2_svm = svm.SVC(C=10, kernel="poly", degree=2)

# Polynomial kernel (degree 3)
poly3_svm = svm.SVC(C=10, kernel="poly", degree=3)

# Fit the models to the dataset
linear_svm.fit(X, y)
poly2_svm.fit(X, y)
poly3_svm.fit(X, y)

models = [linear_svm, poly2_svm, poly3_svm]
titles = ["Linear Kernel", "Poly Kernel (Degree 2)", "Poly Kernel (Degree 3)"]

# Plot the results
plot_results(models, titles, X, y)

# b
################################
# Non-homogeneous polynomial kernel (degree 2)
poly2_non_homogeneous = svm.SVC(C=10, kernel="poly", degree=2, coef0=10)

# Non-homogeneous polynomial kernel (degree 3)
poly3_non_homogeneous = svm.SVC(C=10, kernel="poly", degree=3, coef0=10)

# Fit the models
poly2_non_homogeneous.fit(X, y)
poly3_non_homogeneous.fit(X, y)

models = [poly2_non_homogeneous, poly3_non_homogeneous]
titles = ["Poly Kernel (Degree 2, Non-Homogeneous)", "Poly Kernel (Degree 3, Non-Homogeneous)"]

plot_results(models, titles, X, y)

# c
####################################
# Introduce label noise
flip_indices = (y == -1) & (np.random.random(y.shape) < 0.1)
y_noisy = y.copy()
y_noisy[flip_indices] = 1

poly_svm_noisy = svm.SVC(C=10, kernel="poly", degree=2, coef0=1)
poly_svm_noisy.fit(X, y_noisy)

rbf_svm_noisy = svm.SVC(C=10, kernel="rbf", gamma=10)
rbf_svm_noisy.fit(X, y_noisy)

models_noisy = [poly_svm_noisy, rbf_svm_noisy]
titles_noisy = ["Poly Kernel (Degree 2, Non-Homogeneous)", "RBF Kernel (Gamma = 10)"]

plot_results(models_noisy, titles_noisy, X, y_noisy)

models_noisy = []
titles_nois = []

for i in range(-1, 5):
    gamma = 10 ** i
    rbf_svm_noisy = svm.SVC(C=10, kernel="rbf", gamma=gamma)
    rbf_svm_noisy.fit(X, y_noisy)

    models_noisy=[rbf_svm_noisy]
    titles_noisy=[f"Gamma = {gamma}"]
    plot_results(models_noisy, titles_noisy, X, y_noisy)

