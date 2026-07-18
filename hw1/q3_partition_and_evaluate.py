import numpy as np

# q3a.Data partitioning
def shuffle_and_partition_for_MNIST(data, labels):
    # For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.
    shuffled_indices = np.random.permutation(len(data))
    validation_indices = shuffled_indices[:10000]
    training_indices = shuffled_indices[10000:]
    return data[training_indices], labels[training_indices], data[validation_indices], labels[validation_indices]

def shuffle_and_partition_for_spam(data, labels):
    # For the spam dataset, write code that sets aside 20% of the training data as a validation set.
    shuffled_indices = np.random.permutation(len(data))
    split_index = int(len(data) * 0.8)
    training_indices = shuffled_indices[:split_index]
    validation_indices = shuffled_indices[split_index:]
    return data[training_indices], labels[training_indices], data[validation_indices], labels[validation_indices]

# q3b.Evaluation metric
def evaluate_classifier(true_labels, predicted_labels):
    # Use classification accuracy as the evaluation metric.
    assert len(true_labels) == len(predicted_labels), "Length of true labels and predicted labels must be the same."
    num_correct = np.sum(true_labels == predicted_labels)
    accuracy = num_correct / len(true_labels)
    return accuracy
