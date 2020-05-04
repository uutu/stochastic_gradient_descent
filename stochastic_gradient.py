# Stochastic gradient descent
# ML Exercise 1
# Juuso Kuisma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

np.random.seed(seed=12)

def draw_plot(x, y, w, Error, w_s):
    '''
    A method for drawing the plots of the function.
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28, 10))
    ax[0].plot(x[:, 1], [np.sin(0.8 * np.pi * x_i) for x_i in x[:, 1]], c='red', linewidth=2, zorder=0)
    red_patch = mpatches.Patch(color='red', label='The learned function')
    ax[0].legend(handles=[red_patch])
    ax[0].scatter(x[:, 1], y)
    ax[0].plot(x[:, 1], np.dot(x, w))
    ax[0].set_title('Data points, the sine function, and learned polynomial function')
    ax[1].scatter(range(iterations), Error)
    ax[1].set_title('Error per iteration')

    plt.show()

def generate_data():
    '''
    A method to generate and initialize the data used where the 100 data points take the
    form of a sine wave (4 degrees) and have a slight noise to the distribution.
    '''
    x = PolynomialFeatures(degree=4).fit_transform(np.linspace(-2, 2, 100).reshape(100, -1))
    x[:, 1:] = MinMaxScaler(feature_range=(-2, 2), copy=False).fit_transform(x[:, 1:])
    l = lambda x_i: np.sin(0.8 * np.pi * x_i)
    data = l(x[:, 1])
    # The noise of the distribution from [-0.3, 0-3]
    noise = np.random.normal(0, 0.3, size=np.shape(data))
    y = data + noise
    y = y.reshape(100, 1)
    # Normalize the data and return the generated data points
    return {'x': x, 'y': y}

def stochastic_gradient_descent(x, y, w, learning_rate):
    '''
    The Stochastic Gradient Descent method for solving the regression problem
    '''
    derivative = -(y - np.dot(w.T, x)) * x.reshape(np.shape(w))
    return learning_rate * derivative

# Initialize all of the variables for the function
data = generate_data()
# Tables for both x and y
x = data['x']
y = data['y']
# Initializing the function for the sine wave
w = np.random.normal(size=(np.shape(x)[1], 1))
# Learning rate and number of iterations
learning_rate = 0.001
iterations = 1000

# Update w based on the Stochastic Gradient Descent function
w_s = []
Error = []
for i in range(iterations):

    # Calculation for the error using the root-mean-square error
    error = (1 / 2) * np.sum([(y[i] - np.dot(w.T, x[i, :])) ** 2 for i in range(len(x))])
    # Append the error to the array
    Error.append(error)

    # Stochastic Gradient Descent function call where the given parametres are used
    for d in range(len(x)):
        w -= stochastic_gradient_descent(x[d, :], y[d], w, learning_rate)
        w_s.append(w.copy())

# Show predicted weights by printing them
print(w_s)

# Plot the predicted function and the error and the final learned function
draw_plot(x, y, w, Error, w_s)