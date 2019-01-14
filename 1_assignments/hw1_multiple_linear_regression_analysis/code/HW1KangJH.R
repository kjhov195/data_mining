#HW_1, 2018321086, Jaehun Kang

###you have to change path, data_name, fm!

################################
#1. defining functions
################################

start_fn = function(){
	rm(list=ls())	#clear the old variables
	###you have to change path!
	path = "C:/Users/YISS/Desktop/data_mining/hw/hw1/codes/submitted"
	setwd(path)
}

input_data = function(){
	data_name <<- readline("Enter the data file name: ")
}

read_table = function(){
	cat("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c': ") 
	fm = scan(n=1, quiet=TRUE)
	if(fm==1) {
	  form=""
	} else {
	  form=","
	}
	data1 <<- read.table(data_name, sep=form) 
}

#fitting model
multiple_linear_regression = function(){
	#data: dataframe with no constant column
	n <<- dim(data1)[1]
	p <<- dim(data1)[2]

	#data2: dataframe with constant column
	data2 <<- cbind(constant=1, data1)
	df_Y <<- data2[2]
	df_X <<- data2[-2]

	#matrix
	mat_Y = data.matrix(df_Y)
	mat_X = data.matrix(df_X)
	
	#b_vertor		#b = solve(X_transpose%*%X)%*%X_transpose*%*Y
	mat2 = t(mat_X)
	mat3 = mat2%*%mat_X
	mat4 = solve(mat3)
	mat5 = mat4%*%mat2
	mat6 = mat5%*%mat_Y
	b_vector <<- mat6
	
	len_b <<- length(b_vector)
	
	###prediction
	SLR_model_prediction <<- function(i){
	  y_hat <<- b_vector[1] + sum(df_X[i, -1] * b_vector[-1])
	}

	###evaluation
	mat_diag = diag(n)
	mat_one = matrix(1, nrow=n)
	mat_J = mat_one%*%t(mat_one)
	mat_H = mat_X%*%mat4%*%mat2
	mat_H0 = (1/n)*mat_J

	SSTO <<- t(mat_Y)%*%(mat_diag - mat_H0)%*%mat_Y
	SSE <<- t(mat_Y)%*%(mat_diag - mat_H)%*%mat_Y
	SSR <<- SSTO-SSE

	###R-square
	R_square <<- SSR/SSTO

	###MSE
	MSE <<- SSE/(n-p)
}

#printing result
print_fn = function(){
	cat("Coefficients", "\n", "-------------", "\n",sep="")
	cat("Constant: ", b_vector[1], "\n", sep="")
	for (i in 2:len_b){
	cat("Beta", i-1, ":  ", b_vector[i], "\n", sep="")
	}
	cat("\n")

	cat("ID, Actual values, Fitted values", "\n", "--------------------------------", "\n")
	for (i in 1:n){
		SLR_model_prediction(i)
		cat(i, ", ", data2[i, 2], ", ", y_hat, "\n", sep="") 	
	}
	cat("\n")

	cat("Model Summary", "\n", "-------------", "\n",sep="")
	cat("R-square = ", R_square, "\n")
	cat("MSE = ", MSE)
}	


################################
#2. result
################################

#setting path
start_fn()

#data_name ###you have to change data_name!
input_data()
harris.dat

#raeding table ###you have to change fm = 1 or 2!
read_table()
1

#fitting regression model
multiple_linear_regression()

#saving result
sink('HW1KangJH.txt')
print_fn()
sink()
