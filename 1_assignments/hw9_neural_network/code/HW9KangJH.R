
#HW9KangJH_2018321086

##############################################################
######################R-Code##################################
##############################################################

#immorting modules
install.packages("rgl")
install.packages("nnet")
library(rgl)
library(nnet)

#preprocess
rm(list=ls())
set.seed(2018321086)

#defining functions
sigmoid_fn = function(x){
	temp = 1/(1+exp(-x))
	return(temp)
}

# Generating random numbers
a1 = as.matrix(c(2, 2))
a2 = as.matrix(c(3, -3))
a1_t = t(a1)
a2_t = t(a2)

x1 = rnorm(1000, 0, 1)
x2 = rnorm(1000, 0, 1)
x_data = cbind(x1, x2)
dim(x_data) = c(2, 1000)

h1 = sigmoid_fn(a1_t%*%x_data)
h2 = (a2_t%*%x_data)**2
h3 = as.matrix(0.3*rnorm(1000, 0, 1))
dim(h3) = c(1, 1000)

y_data = h1+h2+h3


#a. plot the surface of responses using function Y = sigmoid(a1_t *%* x_data) + (a2_t %*% x_data_**2)
object_function = function(x,y){
	return(sigmoid_fn(2*x+2*y) + (3*x-3*y)**2)
}
axis_x = seq(-2, 2, length=100)
axis_y = seq(-2, 2, length=100)
axis_z = outer(axis_x, axis_y, object_function)
persp(axis_x, axis_y, axis_z, 
	xlab="axis_x1", ylab="axis_x2", zlab="axis_y", col="gray",
	phi=20, theta=20, expand=1, 
      main = "Surface of Responses Y")


#b. perform a neural network analysis using both R and Python. Fix the number of hidden layer as one. Use other options as default.
x1_test = rnorm(1000, 0, 1)
x2_test = rnorm(1000, 0, 1)
x_data_test = cbind(x1_test, x2_test)
dim(x_data_test) = c(2, 1000)
h1_test = sigmoid_fn(a1_t%*%x_data_test)
h2_test = (a2_t%*%x_data_test)**2
h3_test = as.matrix(0.3*rnorm(1000, 0, 1))
dim(h3_test) = c(1, 1000)
y_data_test = h1_test+h2_test+h3_test

neuralnet_model = nnet(t(x_data), t(y_data), size = 1, linout = TRUE)
y_data_predict = neuralnet_model$fitted #predicted value
training_error = sum((y_data_predict - t(y_data))**2)/1000
testing_error = sum((y_data_predict - t(y_data_test))**2)/1000

sprintf("training_error = %f, testing_error = %f", training_error, testing_error) 


#c. vary the number of hidden nodes in the hidden layer from 2 up to 10.
list_predict = c()
list_training_error = c()
list_testing_error = c()

for (i in 1:10){
	cat("the number of hidden nodes: ", i, "\n")
	temp_nn_model = nnet(t(x_data), t(y_data), size = i, linout = TRUE)
	list_predict = cbind(list_predict, temp_nn_model$fitted.value)
	list_training_error = cbind(list_training_error, sum((as.data.frame(list_predict[,i])-t(y_data))**2)/1000)
	list_testing_error = cbind(list_testing_error, sum((predict(temp_nn_model,t(x_data_test))-t(y_data_test))**2)/1000)
}

list_training_error = as.numeric(list_training_error)
list_testing_error = as.numeric(list_testing_error)

result_c = cbind(as.data.frame(list_training_error), as.data.frame(list_testing_error))
result_c


#d. plot the training and test error curves as a function of the number of hidden nodes.
plot(list_training_error,type='b',lty=1,ylim=c(min(list_training_error,list_testing_error),max(list_training_error,list_testing_error)),
     xlab="Number of hidden nodes",ylab="Error",col="blue")
lines(list_testing_error,type='b',lty=2,col="red")


#e. determine the minimum # of hidden nodes needed to perform well for this task.
list_testing_error
min(list_testing_error)