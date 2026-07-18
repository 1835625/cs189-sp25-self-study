import sklearn
import numpy as np
import matplotlib.pyplot as plt

def shuffle_and_partition(data, labels):
    # For the spam dataset, write code that sets aside 20% of the training data as a validation set.
    shuffled_indices = np.random.permutation(len(data))
    split_index = int(len(data) * 0.8)
    training_indices = shuffled_indices[:split_index]
    validation_indices = shuffled_indices[split_index:]
    return data[training_indices], labels[training_indices], data[validation_indices], labels[validation_indices]

def train_svm_classifier(training_data, training_labels):
    clf = sklearn.svm.SVC(kernel = 'linear')
    clf.fit(training_data, training_labels)
    return clf

def evaluate_classifier(true_labels, predicted_labels):
    # Use classification accuracy as the evaluation metric.
    assert len(true_labels) == len(predicted_labels), "Length of true labels and predicted labels must be the same."
    num_correct = np.sum(true_labels == predicted_labels)
    accuracy = num_correct / len(true_labels)
    return accuracy

if __name__ == "__main__":
    # Load the spam dataset.
    data = np.load("data/spam-data.npz")
    training_data, training_labels, validation_data, validation_labels = shuffle_and_partition(data["training_data"], data["training_labels"])

    # Train the SVM classifier with different numbers of training examples.
    num_examples_list = [100, 200, 500, 1000, 2000, len(training_data)]
    accuracy_list = []
    for num_examples in num_examples_list:
        clf = train_svm_classifier(training_data[:num_examples], training_labels[:num_examples])

        # Predict the labels for validation data and evaluate the classifier.
        predicted_labels = clf.predict(validation_data)
        accuracy = evaluate_classifier(validation_labels, predicted_labels)
        accuracy_list.append(accuracy)

    # Plot accuracy versus number of training examples.
    plt.plot(num_examples_list, accuracy_list, marker='o')
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Classifier Accuracy vs. Number of Training Examples")
    plt.show()