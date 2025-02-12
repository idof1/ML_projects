from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']


def e_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def k_NN(train_images, labels, query_image, k):
    distances = np.array([e_distance(image, query_image) for image in train_images])

    k_near_images = np.argpartition(distances, k)[:k]

    max_l = 0, labels[k_near_images[0]]
    label_cnt = {}
    for i in k_near_images:
        label = labels[i]
        if label in label_cnt:
            label_cnt[label] += 1
        else:
            label_cnt[label] = 1
        if label_cnt[label] > max_l[0]:
            max_l = label_cnt[label], label

    return max_l[1]

# def accuracy(predictions, test_labels):
#     return (np.array(predictions) == np.array(test_labels)).mean() * 100
#
# n = 1000  # Number of training images
# k = 10  # Number of nearest neighbors
# train_images = data[:n]  # First 1000 images as training set
# train_labels = labels[:n]  # Corresponding labels
#
# # Test the algorithm on each of the test images
# test_images = data[n:2 * n]  # Next 1000 images as test set
# test_labels = labels[n:2 * n]  # Corresponding labels for testing
#
# # Run k-NN on each test image and collect results
# predictions = []
# for query_image in test_images:
#     predicted_label = k_NN(train_images, train_labels, query_image, k)
#     predictions.append(predicted_label)
#
# # Calculate accuracy
# correct_predictions = sum(pred == true for pred, true in zip(predictions, test_labels))
# accuracy = (correct_predictions / len(test_labels)) * 100
# print("Accuracy: " + str(accuracy) + "%")
#
# # Test accuracy for k = 1 to 100
# accuracy_results = []
# for k in range(1, 101):
#     predictions = [k_NN(train_images, train_labels, test_image, k) for test_image in test_images]
#     correct_predictions = sum(pred == true for pred, true in zip(predictions, test_labels))
#     accuracy = (correct_predictions / len(test_labels)) * 100
#     accuracy_results.append(accuracy)
#
# # Plot accuracy as a function of k
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 101), accuracy_results, marker='o')
# plt.xlabel("k (Number of Neighbors)")
# plt.ylabel("Accuracy (%)")
# plt.title("k-NN Accuracy as a Function of k")
# plt.grid(True)
# plt.show()
#
# # Find the best k
# best_k = np.argmax(accuracy_results) + 1  # Add 1 because index starts from 0
# best_accuracy = accuracy_results[best_k - 1]
# print(f"Best k: {best_k} with accuracy: {best_accuracy:.2f}%")

# predictions = []
# accuracy_results = []
# for i in range(100, 5100, 100):
#     n = i  # Number of training images
#     k = 1  # Number of nearest neighbors
#     train_images = data[:n]  # First 1000 images as training set
#     train_labels = labels[:n]  # Corresponding labels
#
#     # Test the algorithm on each of the test images
#     test_images = data[n:2 * n]  # Next 1000 images as test set
#     test_labels = labels[n:2 * n]  # Corresponding labels for testing
#
#     predictions = [k_NN(train_images, train_labels, test_image, k) for test_image in test_images]
#     accuracy_results.append(accuracy(predictions, test_labels))
#
# # Plot accuracy as a function of n
# plt.figure(figsize=(10, 6))
# plt.plot(range(100, 5100,100), accuracy_results, marker='o')
# plt.xlabel("n (Number of Images)")
# plt.ylabel("Accuracy (%)")
# plt.title("1-NN Accuracy as a Function of n")
# plt.grid(True)
# plt.show()
