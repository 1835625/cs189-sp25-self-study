import sklearn
import numpy as np
import matplotlib.pyplot as plt

def shuffle_and_partition(data, labels):
    # For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.
    shuffled_indices = np.random.permutation(len(data))
    validation_indices = shuffled_indices[:10000]
    training_indices = shuffled_indices[10000:]
    return data[training_indices], labels[training_indices], data[validation_indices], labels[validation_indices]

def train_svm_classifier(training_data, training_labels):
    # Train a linear SVM classifier using the training data and labels.
    clf = sklearn.svm.SVC(kernel='linear')
    clf.fit(training_data, training_labels)
    return clf

def evaluate_classifier(true_labels, predicted_labels):
    # Use classification accuracy as the evaluation metric.
    assert len(true_labels) == len(predicted_labels), "Length of true labels and predicted labels must be the same."
    num_correct = np.sum(true_labels == predicted_labels)
    accuracy = num_correct / len(true_labels)
    return accuracy

if __name__ == "__main__":
    # Load the MNIST dataset.
    data = np.load("data/mnist-data.npz")
    training_data, training_labels, validation_data, validation_labels = shuffle_and_partition(data["training_data"], data["training_labels"])
    training_data = (
        training_data[:10000]
        .reshape(-1, 28 * 28)
        .astype(np.float32) / 255.0
    )
    validation_data = (
        validation_data
        .reshape(-1, 28 * 28)
        .astype(np.float32) / 255.0
    )

    # Train the SVM classifier with different values of regularization parameter C.
    C_values = np.logspace(-4, 3, 8)
    accuracy_list = []
    for C in C_values:
        # Train the SVM classifier with the current value of C.
        clf = sklearn.svm.SVC(kernel = 'linear', C = C)
        clf.fit(training_data, training_labels[:10000])

        # Predict the labels for validation data and evaluate the classifier.
        predicted_labels = clf.predict(validation_data)
        accuracy = evaluate_classifier(validation_labels, predicted_labels)
        print(f"Validation accuracy for C={C}: {accuracy:.4f}")
        accuracy_list.append(accuracy)
    plt.plot(C_values, accuracy_list, marker='o')
    plt.xscale('log')
    plt.xlabel("Regularization Parameter C (log scale)")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Classifier Accuracy vs. Regularization Parameter C")
    plt.show()