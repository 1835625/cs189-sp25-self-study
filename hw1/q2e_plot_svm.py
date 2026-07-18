import numpy as np
import matplotlib.pyplot as plt

# Plot the data points
def plot_data_points(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels)

# Plot the decision boundary
def plot_decision_boundary(w, b):
    x = np.linspace(-5, 5, 100)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, 'k')
         
# Plot the margins
def plot_margins(w, b):
    x = np.linspace(-5, 5, 100)
    y1 = (1 - b - w[0] * x) / w[1]
    y2 = (-1 - b - w[0] * x) / w[1]
    plt.plot(x, y1, 'k--')
    plt.plot(x, y2, 'k--')

if __name__ == "__main__":
    data = np.load("data/toy-data.npz")
    w = [-0.4528, -0.5190]
    b = 0.1471
    plot_data_points(data["training_data"], data["training_labels"])
    plot_decision_boundary(w, b)
    plot_margins(w, b)
    plt.show()
