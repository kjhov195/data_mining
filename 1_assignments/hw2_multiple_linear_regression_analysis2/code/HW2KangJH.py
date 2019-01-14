#HW_2, 2018321086, Jaehun Kang

import os
import numpy as np
import pandas as pd

################################
#1. defining functions
################################

def start_fn():
	global path
	path = "C:/Users/YISS/Desktop/data_mining/hw/hw2"
	os.chdir(path)

def input_data():
	global data_name
	print("Enter the data file name: ")
	data_name = input()

def read_table():
	global form
	global data1
	print("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c': ")
	fm = int(input())
	if fm==1:
		form = " "
	else:
		form = ","
	data1 = pd.read_csv(data_name, sep=form, header=None)

def response_var_col():
	global col_num
	col_num = int(input())

#fitting model
def multiple_linear_regression():
	global n, m, data2, df_Y, df_X, b_vector, len_b, R_square, MSE, SLR_model_prediction

	#data: dataframe with no constant column
	n = data1.shape[0]
	p = data1.shape[1]

	#data2: dataframe with constant column
	temp1 = np.ones(shape=(n,1), dtype=int)
	temp2 = np.array(data1)

	temp3 = np.concatenate((temp1, temp2), axis=1)
	data2 = pd.DataFrame(temp3)

	df_Y = data2.iloc[ :, [col_num]]
	temp4 = data2
	df_X = temp4.drop(temp4.columns[col_num], axis=1)

	#matrix
	mat_Y = np.array(df_Y)
	mat_X = np.array(df_X)

	#b_vertor		#b = solve(X_transpose%*%X)%*%X_transpose*%*Y
	mat2 = np.transpose(mat_X)
	mat3 = np.matmul(mat2, mat_X)
	mat4 = np.linalg.inv(mat3)
	mat5 = np.matmul(mat4, mat2)
	mat6 = np.matmul(mat5, mat_Y)
	b_vector = mat6

	len_b = len(b_vector)

	###prediction
	def SLR_model_prediction(i):
		global y_hat
		aa= np.array(df_X.iloc[i, :])
		bb= b_vector.reshape(p, )
		y_hat = sum(aa*bb)

	###evaluation
	mat_diag = np.eye(n)
	mat_one = np.ones(shape=(n, 1))
	mat_J = np.matmul(mat_one, np.transpose(mat_one))
	mat_H = np.matmul(np.matmul(mat_X, mat4), mat2)
	mat_H0 = (1/n)*mat_J

	SSTO = np.matmul(np.matmul(np.transpose(mat_Y), (mat_diag - mat_H0)), mat_Y)
	SSE = np.matmul(np.matmul(np.transpose(mat_Y), (mat_diag - mat_H)), mat_Y)
	SSR = SSTO-SSE

	SSTO = SSTO[0][0]
	SSE = SSE[0][0]
	SSR = SSR[0][0]

	###R-square
	R_square = SSR/SSTO

	###MSE
	MSE = SSE/(n-p)

#printing result
def print_fn():
	print("Coefficients", "\n", "-------------",sep="")
	print("Constant: ", b_vector[0][0], sep="")
	for j in range(0, len_b-1):
		print("Beta", j+1, ":  ", b_vector[j+1][0], sep="")
	print("")

	print("ID, Actual values, Fitted values", "\n", "--------------------------------")
	for k in range(n):
		SLR_model_prediction(k)
		print(k+1, ", ", data2.iloc[k, col_num], ", ", y_hat, sep="")
	print("")

	print("Model Summary", "\n", "-------------",sep="")
	print("R-square: ", R_square)
	print("MSE: ", MSE)

#save
def save_as_file():
	with open("HW2KangJH_python_output.txt","w") as txt:
		print("Coefficients", "\n", "-------------",sep="", file=txt)
		print("Constant: ", b_vector[0][0], sep="", file=txt)
		for j in range(0, len_b-1):
			print("Beta", j+1, ":  ", b_vector[j+1][0], sep="", file=txt)
		print("", file=txt)

		print("ID, Actual values, Fitted values", "\n", "--------------------------------", file=txt)
		for k in range(n):
			SLR_model_prediction(k)
			print(k+1, ", ", data2.iloc[k, col_num], ", ", y_hat, sep="", file=txt)
		print("", file=txt)

		print("Model Summary", "\n", "-------------",sep="", file=txt)
		print("R-square: ", R_square, file=txt)
		print("MSE: ", MSE, file=txt)

################################
#2. result
################################
#setting path
start_fn()

#data_name
input_data()

#raeding table
read_table()

#which column the response variable is recorded
response_var_col()

#fitting regression model
multiple_linear_regression()

#printing result
print_fn()

#save
save_as_file()
