import sklearn
import numpy as np
import matplotlib.pyplot as plt

def shuffle_and_partition_into_5_folds(data, labels):
    # Shuffle the data.
    shuffled_indices = np.random.permutation(len(data))

    # Partition the data into 5 folds.
    fold_size = len(data) // 5
    folds = []
    for i in range(5):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size if i < 4 else len(data)
        fold_indices = shuffled_indices[start_index:end_index]
        folds.append((data[fold_indices], labels[fold_indices]))
    return folds

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

    # Shuffle and partition the data into 5 folds.
    folds = shuffle_and_partition_into_5_folds(data["training_data"], data["training_labels"])

    # Train and evaluate the SVM classifier using 5-fold cross-validation with different values of regularization parameter C.
    C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    accuracy_list = []

    for C in C_values:
        fold_accuracies = []
        for i in range(5):
            # Prepare training and validation data for the current fold.
            validation_data, validation_labels = folds[i]
            training_data = np.concatenate([folds[j][0] for j in range(5) if j != i])
            training_labels = np.concatenate([folds[j][1] for j in range(5) if j != i])

            # Train the SVM classifier with the current value of C.
            clf = sklearn.svm.SVC(kernel = 'linear', C = C)
            clf.fit(training_data, training_labels)

            # Predict the labels for validation data and evaluate the classifier.
            predicted_labels = clf.predict(validation_data)
            accuracy = evaluate_classifier(validation_labels, predicted_labels)
            fold_accuracies.append(accuracy)
        
        # Calculate the average accuracy across all folds for the current value of C.
        average_accuracy = np.mean(fold_accuracies)
        accuracy_list.append(average_accuracy)
        print(f"Average validation accuracy for C={C}: {average_accuracy:.4f}")

    # Plot average accuracy versus regularization parameter C.
    plt.plot(C_values, accuracy_list, marker='o')
    plt.xscale('log')
    plt.xlabel("Regularization Parameter C (log scale)")
    plt.ylabel("Average Validation Accuracy")
    plt.title("SVM Classifier Average Accuracy vs. Regularization Parameter C (5-Fold Cross-Validation)")
    plt.show()