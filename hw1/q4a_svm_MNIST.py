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

    # Train the SVM classifier with different numbers of training examples.
    num_examples_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    accuracy_list = []
    for num_examples in num_examples_list:
        # Resize the training data from 28*28 to 784 and train the classifier.
        training_data_reshaped = training_data[:num_examples].reshape(-1, 28 * 28)
        clf = train_svm_classifier(training_data_reshaped, training_labels[:num_examples])

        # Predict the labels for validation data and evaluate the classifier.
        validation_data_reshaped = validation_data.reshape(-1, 28 * 28)
        predicted_labels = clf.predict(validation_data_reshaped)
        accuracy = evaluate_classifier(validation_labels, predicted_labels)
        accuracy_list.append(accuracy)
    
    # Plot accuracy versus number of training examples.
    plt.plot(num_examples_list, accuracy_list, marker='o')
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Classifier Accuracy vs. Number of Training Examples")
    plt.show()
