import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.neural_network import MLPRegressor

random.seed(2018321086)

a1 = np.array([2,2]).reshape(2,1)
a2 = np.array([3,-3]).reshape(2,1)
a1_t = np.transpose(a1)
a2_t = np.transpose(a2)

x1 = np.random.normal(0, 1, 1000).reshape(1, 1000)
x2 = np.random.normal(0, 1, 1000).reshape(1, 1000)
x_data = np.concatenate((x1, x2), axis=0)

h1 = (1/(1+np.exp(-np.matmul(a1_t, x_data))))
h2 = (np.matmul(a2_t, x_data))**2
h3 = 0.3*np.random.normal(0, 1, 1000).reshape(1, 1000)
y_data = h1+h2+h3


#a. plot the surface of responses using function Y = sigmoid(a1_t *%* x_data) + (a2_t %*% x_data_**2)
x1_to_plot = np.repeat(np.arange(-2, 2, 0.05), [80]*80)
x2_to_plot = np.tile(np.arange(-2, 2, 0.05), 80)
y_data_to_plot = (1/(1+np.exp(-(2*x1_to_plot + 2*x2_to_plot)))) + (3*x1_to_plot + -3*x2_to_plot)**2

plot1 = plt.figure()
plot2 = plot1.add_subplot(111, projection='3d')
plot2.scatter(x1_to_plot, x2_to_plot, y_data_to_plot, c='b', marker=".")
plot2.set_xlabel('x1')
plot2.set_ylabel('x2')
plot2.set_zlabel('y')
plt.show()


#b. perform a neural network analysis using both R and Python. Fix the number of hidden layer as one. Use other options as default.
x1_test = np.random.normal(0, 1, 1000).reshape(1, 1000)
x2_test = np.random.normal(0, 1, 1000).reshape(1, 1000)
x_data_test = np.concatenate((x1, x2), axis=0)

h1_test = (1/(1+np.exp(-np.matmul(a1_t, x_data_test))))
h2_test = (np.matmul(a2_t, x_data_test))**2
h3_test = 0.3*np.random.normal(0, 1, 1000).reshape(1, 1000)
y_data_test = h1_test+h2_test+h3_test

x_data_transpose = np.transpose(x_data)
y_data_transpose = np.transpose(y_data)
x_data_test_transpose = np.transpose(x_data_test)
y_data_test_transpose = np.transpose(y_data_test)

model_neuralnet = MLPRegressor(hidden_layer_sizes=(1), solver='lbfgs')
model_neuralnet.fit(x_data_transpose, y_data_transpose)

training_error = sum(sum((model_neuralnet.predict(x_data_transpose)-y_data)**2))/1000
testing_error = sum(sum((model_neuralnet.predict(x_data_transpose)-y_data_test)**2))/1000

print("training_error = %f, testing_error = %f"%(training_error, testing_error))


#c. vary the number of hidden nodes in the hidden layer from 2 up to 10.
list_training_error = []
list_testing_error = []

for i in range(1,11):
    model_neuralnet_ith = MLPRegressor(hidden_layer_sizes=(i), solver='lbfgs')
    model_neuralnet_ith.fit(x_data_transpose, y_data_transpose)
    training_error = sum(sum((model_neuralnet_ith.predict(x_data_transpose)-y_data)**2))/1000
    testing_error = sum(sum((model_neuralnet_ith.predict(x_data_test_transpose)-y_data_test)**2))/1000
    list_training_error.append(training_error)
    list_testing_error.append(testing_error)

result_c = pd.DataFrame(np.concatenate((np.array(list_training_error).reshape(10, 1), np.array(list_testing_error).reshape(10, 1)), axis=1), columns=['training', 'testing'])
print(result_c)


#d. plot the training and test error curves as a function of the number of hidden nodes.
x_axis = range(1, 11)
plt.figure()
plt.plot(x_axis, list_training_error,'b--')
plt.plot(x_axis, list_testing_error, 'r--')
plt.xlabel('# of hidden nodes')
plt.ylabel('error')
plt.show()


#e. determine the minimum # of hidden nodes needed to perform well for this task.
list_testing_error
np.argmin(list_testing_error)
