import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

################################
#1. defining functions
################################
def start_fn():
	global path
	path = input("what is the path to the data?")#C:/Users/YISS/Desktop/data_mining/hw/hw8
	os.chdir(path)

def which_analysis():
	global analysis
	print("which analysis do you want?(1 = regression or 2 = classification)")
	analysis = int(input("which analysis do you want?(1 = regression or 2 = classification)"))

def reg_or_cla():
	start_fn()
	which_analysis()
	if analysis==1:
		input_data()
		read_table()
		response_var_col()
		multiple_linear_regression()
		print_fn_reg()
		save_as_file_reg()
	elif analysis==2:
		training_data()
		read_table_training()
		testing_data()
		read_table_test()
		class_col()
		classification_data_preprocessing()
		classification()
		get_confusion_matrix()
		save_as_file_classification()
	else:
		print("you typed wrong. try again.")

################################
#1-1. multiple_linear_regression
################################
def input_data():
	global data_name
	print("Enter the data file name: ")
	data_name = input("Enter the data file name: ")

def read_table():
	global form
	global data1
	print("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c': )")
	fm = int(input("Select the data coding format(1 = 'a b c' or 2 = 'a,b,c': )"))
	if fm==1:
		form = " "
	else:
		form = ","
	data1 = pd.read_csv(data_name, sep=form, header=None)

def response_var_col():
	global col_num
	print("Select the column num of response variable(ex. 1, 2, 3, ...)")
	col_num = int(input("Select the column num of response variable(ex. 1, 2, 3, ...)"))

#fitting model
def multiple_linear_regression():
	global n, p, m, data2, df_Y, df_X, b_vector, len_b, R_square, MSE, SLR_model_prediction
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
def print_fn_reg():
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
def save_as_file_reg():
	with open("HW8KangJH_python_regression_output.txt","w") as txt:
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
#1-2. classification(LDA, QDA, RDA, Logistic)
################################
def training_data():
	global data_name
	print("Enter the training data file name: ")
	data_name = input("Enter the training data file name: ")

def testing_data():
	global data_name2
	print("Enter the testing data file name: (if the test data set doesn't exist, enter the training data file name)")
	data_name2 = input("Enter the testing data file name: (if the test data set doesn't exist, enter the training data file name)")

def read_table_training():
	global form
	global data1
	print("Select the training data coding format(1 = 'a b c' or 2 = 'a,b,c': )")
	fm = int(input("Select the training data coding format(1 = 'a b c' or 2 = 'a,b,c': )"))
	if fm==1:
		form = " "
	else:
		form = ","
	data1 = pd.read_csv(data_name, sep=form, header=None)

def read_table_test():
	global form
	global data1_test
	print("Select the testing data coding format(1 = 'a b c' or 2 = 'a,b,c': )")
	fm = int(input("Select the testing data coding format(1 = 'a b c' or 2 = 'a,b,c': )"))
	if fm==1:
		form = " "
	else:
		form = ","
	data1_test = pd.read_csv(data_name2, sep=form, header=None)

def class_col():
	global col_num
	print("Select the number of the class column(ex. 1, 2, 3, ...)")
	col_num = int(input("Select the number of the class column(ex. 1, 2, 3, ...)"))
	col_num -= 1

def classification_data_preprocessing():
	global n, n2, p, p2, data1, data1_test, list_classes, list_classes_cnt, K, list_p, list_S, Sp, Sp_inv, data11, data11_test, col_name
	try:
		data1 = data1.replace('?', np.nan)
		data11 = data1
		data1 = data1.apply(pd.to_numeric)
		data1_test = data1_test.replace('?', np.nan)
		data11_test = data1_test
		data1_test = data1_test.apply(pd.to_numeric)
		n = data1.shape[0]
		p = data1.shape[1]
		n2 = data1_test.shape[0]
		p2 = data1_test.shape[1]
		list_classes = data1[[col_num]].groupby(col_num).size().index.tolist()
		list_classes_cnt = data1[[col_num]].groupby(col_num).size().tolist()
		#K : number of classes
		K = len(list_classes)
		#p(w_k)
		list_p = np.array(list_classes_cnt)/n
		#Sp
		list_S = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[col_num]==list_classes[k-1]]
			data1_k_x = data1_k.drop([col_num], axis=1)
			list_xk_bar = np.array(pd.DataFrame.sum(data1_k_x, axis=0)).reshape(p-1,1)/n_k
			list_tempk = []
			for i in range(n_k):
				list_xk_temp = np.array(data1_k_x.iloc[i, :]).reshape(p-1, 1)
				mat7 = list_xk_temp - list_xk_bar
				temp5 = np.matmul(mat7, np.transpose(mat7))
				list_tempk.append(temp5)
			S_k = sum(list_tempk)/(n_k-1)
			list_S.append(S_k)
		temp6 = 0
		for i in range(len(list_S)):
			temp6 += (list_classes_cnt[i]-1)*list_S[i]
		Sp = 1/(sum(list_classes_cnt)-K)*(temp6)
		Sp_inv = np.linalg.inv(Sp)
	except ValueError:
		data1 = pd.read_csv(data_name, sep=form, header=0)
		data1_test = pd.read_csv(data_name2, sep=form, header=0)
		n = data1.shape[0]
		p = data1.shape[1]
		n2 = data1_test.shape[0]
		p2 = data1_test.shape[1]
		col_name = data1.columns[col_num]
		list_classes = data1[[col_name]].groupby(col_name).size().index.tolist()
		list_classes_cnt = data1[[col_name]].groupby(col_name).size().tolist()
		K = len(list_classes)

def classification_LDA():
	global data1, data1_x, data1_test, data1_test_x, accuracy_LDA_training, accuracy_LDA_testing, list_output_class1, list_output_class2, name
	name = "LDA"
	data1 = np.array(data1)
	data1_test = np.array(data1_test)
	data1_x = np.delete(data1, [p-1], axis=1)
	data1_test_x = np.delete(data1_test, [p-1], axis=1)

	#training_data
	list_output_class1 = []
	for i in range(len(data1_x)):
		x_vec = data1_x[i, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			d_k_x = np.matmul(np.matmul(np.transpose(list_xk_bar), Sp_inv), x_vec)- 1/2*(np.matmul(np.matmul(np.transpose(list_xk_bar), Sp_inv), list_xk_bar)) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class1 = list_discriminant.index(max(list_discriminant))+1
		list_output_class1.append(output_class1)
	accuracy_LDA_training = sum(data1[:, p-1] == list_output_class1)/len(data1)

	#testing_data
	list_output_class2 = []
	for i in range(len(data1_test_x)):
		x_vec = data1_test_x[i, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			d_k_x = np.matmul(np.matmul(np.transpose(list_xk_bar), Sp_inv), x_vec)- 1/2*(np.matmul(np.matmul(np.transpose(list_xk_bar), Sp_inv), list_xk_bar)) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class2 = list_discriminant.index(max(list_discriminant))+1
		list_output_class2.append(output_class2)
	accuracy_LDA_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)

def classification_QDA():
	global data1, data1_x, data1_test, data1_test_x, accuracy_QDA_training, accuracy_QDA_testing, list_output_class1, list_output_class2, name
	name = "QDA"
	data1 = np.array(data1)
	data1_test = np.array(data1_test)
	data1_x = np.delete(data1, [p-1], axis=1)
	data1_test_x = np.delete(data1_test, [p-1], axis=1)

	#training_data
	list_output_class1 = []
	for i in range(len(data1_x)):
		x_vec = data1_x[i, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			d_k_x = -1/2*np.log(np.linalg.det(list_S[k-1]))- 1/2*np.matmul(np.matmul(np.transpose( x_vec - list_xk_bar ), np.linalg.inv(list_S[k-1])), ( x_vec - list_xk_bar )) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class1 = list_discriminant.index(max(list_discriminant))+1
		list_output_class1.append(output_class1)
	accuracy_QDA_training = sum(data1[:, p-1] == list_output_class1)/len(data1)

	#testing_data
	list_output_class2 = []
	for i in range(len(data1_test_x)):
		x_vec = data1_test_x[i, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			d_k_x = -1/2*np.log(np.linalg.det(list_S[k-1]))- 1/2*np.matmul(np.matmul(np.transpose( x_vec - list_xk_bar ), np.linalg.inv(list_S[k-1])), ( x_vec - list_xk_bar )) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class2 = list_discriminant.index(max(list_discriminant))+1
		list_output_class2.append(output_class2)
	accuracy_QDA_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)

def classification_RDA():
	global data1, data1_x, data1_test, data1_test_x, accuracy_RDA_training, accuracy_RDA_testing, list_output_class1, list_output_class2, alpha, gamma, list_RDA_2, name
	name = "RDA"
	#getting alpha, gamma
	data1 = np.array(data1)
	data1_test = np.array(data1_test)
	data1_x = np.delete(data1, [p-1], axis=1)
	data1_test_x = np.delete(data1_test, [p-1], axis=1)
	list_RDA_1 = np.array([0, 0, 0]).reshape(1, 3)
	mat_sigma = np.identity(p-1)*(sum(np.diag(Sp))/len(np.diag(Sp)))
	alpha = -0.05
	for i in range(21):
		alpha += 0.05
		gamma = -0.05
		for j in range(21):
			gamma += 0.05
			#testing_data
			list_output_class2 = []
			for h in range(len(data1_test_x)):
				x_vec = data1_test_x[h, :].reshape(p-1, 1)
				list_discriminant = []
				for k in range(1, K+1):
					n_k = list_classes_cnt[k-1]
					data1_k = data1[data1[:, p-1] == k]
					data1_k_x = np.delete(data1_k, [p-1], axis=1)
					list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
					mat8 = x_vec - list_xk_bar
					Sk_alpha_gamma = alpha*list_S[k-1] + (1-alpha)*(gamma*Sp + (1-gamma)*mat_sigma)
					d_k_x = -1/2*np.log(np.linalg.det(Sk_alpha_gamma))- 1/2*np.matmul(np.matmul(np.transpose(mat8), np.linalg.inv(Sk_alpha_gamma)), (mat8)) + np.log(list_p[k-1])
					list_discriminant.append(d_k_x[0][0])
				output_class2 = list_discriminant.index(max(list_discriminant))+1
				list_output_class2.append(output_class2)
			accuracy_RDA_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)
			list_RDA_temp = np.array([alpha, gamma, accuracy_RDA_testing]).reshape(1, 3)
			list_RDA_1 = np.concatenate((list_RDA_1, list_RDA_temp), axis=0)
			print("Calculating..: ", round(((21*i+j)/(21*21))*100, 2), "%", sep="")
	print("Calculating..: ", 100, "%", "\nComplete!", sep="")
	list_RDA_2 = np.round(list_RDA_1[1:], 4)
	alpha = list_RDA_2[list_RDA_2[:, 2]==max(list_RDA_2[:, 2])][0][0]
	gamma = list_RDA_2[list_RDA_2[:, 2]==max(list_RDA_2[:, 2])][0][1]
	#training
	list_RDA_3 = np.array([0, 0, 0, 0]).reshape(1, 4)
	list_output_class1 = []
	for h in range(len(data1_x)):
		x_vec = data1_x[h, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			mat8 = x_vec - list_xk_bar
			Sk_alpha_gamma = alpha*list_S[k-1] + (1-alpha)*(gamma*Sp + (1-gamma)*mat_sigma)
			d_k_x = -1/2*np.log(np.linalg.det(Sk_alpha_gamma))- 1/2*np.matmul(np.matmul(np.transpose(mat8), np.linalg.inv(Sk_alpha_gamma)), (mat8)) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class1 = list_discriminant.index(max(list_discriminant))+1
		list_output_class1.append(output_class1)
	accuracy_RDA_training = sum(data1[:, p-1] == list_output_class1)/len(data1)
	#testing
	list_output_class2 = []
	for h in range(len(data1_test_x)):
		x_vec = data1_test_x[h, :].reshape(p-1, 1)
		list_discriminant = []
		for k in range(1, K+1):
			n_k = list_classes_cnt[k-1]
			data1_k = data1[data1[:, p-1] == k]
			data1_k_x = np.delete(data1_k, [p-1], axis=1)
			list_xk_bar = data1_k_x.sum(axis=0).reshape(p-1,1)/n_k
			mat8 = x_vec - list_xk_bar
			Sk_alpha_gamma = alpha*list_S[k-1] + (1-alpha)*(gamma*Sp + (1-gamma)*mat_sigma)
			d_k_x = -1/2*np.log(np.linalg.det(Sk_alpha_gamma))- 1/2*np.matmul(np.matmul(np.transpose(mat8), np.linalg.inv(Sk_alpha_gamma)), (mat8)) + np.log(list_p[k-1])
			list_discriminant.append(d_k_x[0][0])
		output_class2 = list_discriminant.index(max(list_discriminant))+1
		list_output_class2.append(output_class2)
	accuracy_RDA_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)
	#result
	list_RDA_temp2 = np.array([alpha, gamma, accuracy_RDA_training, accuracy_RDA_testing]).reshape(1, 4)

def sigmoid_fn(xb):
	return(1/(1+np.exp(-xb)))

def minus_log_likelyhood(beta_vec):
	global log_likelyhood
	part_left = np.matmul(np.transpose(input_y), np.matmul(input_x, beta_vec))
	part_right = sum(np.log(1/(1+np.exp(np.matmul(input_x, beta_vec)))))
	likelyhood = np.exp(part_left+part_right)
	log_likelyhood = np.log(likelyhood)
	return((-1)*log_likelyhood)

def mle_maximize(x, y):
	global result_minimize, input_x, input_y
	input_x = x
	input_y = y
	result_minimize = minimize(minus_log_likelyhood, np.zeros((p, 1))) #it can occur error due to the diffusion of 'likelyhood'

def classification_logistic():
	global data1, data1_x, data1_test, data1_test_x, data1_y, data1_test_y, accuracy_logistic_training, accuracy_logistic_testing, list_output_class1, list_output_class2, beta_vec, predicted_result1, predicted_result2, name
	name = "logistic"
	threshold = 0.5

	data1 = np.array(data1)
	data1_x = np.delete(data1, [p-1], axis=1)
	data1_y = data1[:, p-1]
	data1_test = np.array(data1_test)
	data1_test_x = np.delete(data1_test, [p-1], axis=1)
	data1_test_y = data1_test[:, p-1]
	n2 = data1_test.shape[0]

	class_1 = int(max(data1_y))
	class_0 = int(min(data1_y))

	temp_ones = np.ones(n, dtype=int).reshape(n,1)
	temp_ones2 = np.ones(n2, dtype=int).reshape(n2,1)
	data2_x = np.concatenate((temp_ones, data1_x), axis=1)
	data2_y = ((data1_y==class_1)*1).reshape(n,1) #class value값이 max값(2)를 1로, 1을 0으로.
	data2_test_x = np.concatenate((temp_ones2, data1_test_x), axis=1)
	data2_test_y = ((data1_test_y==class_1)*1).reshape(n2,1) #class value값이 max값(2)를 1로, 1을 0으로.

	#training
	mle_maximize(data2_x, data2_y) #it can occur error due to the diffusion of 'likelyhood'
	beta_vec = result_minimize.x.reshape(p, 1)
	xb = np.matmul(data2_x, beta_vec)
	predicted_result1 = sigmoid_fn(xb)
	result_class = (predicted_result1 > threshold)*1
	list_output_class1 = []
	for i in range(len(result_class)):
		if result_class[i] == 1:
			list_output_class1.append(class_1)
		elif result_class[i] == 0:
			list_output_class1.append(class_0)
		else:
			print("error in 'result_class' in %d data!!"%i)
	accuracy_logistic_training = sum(data1[:, p-1] == list_output_class1)/len(data1)
	#testing
	xb = np.matmul(data2_test_x, beta_vec)
	predicted_result2 = sigmoid_fn(xb)
	result_class = (predicted_result2 > threshold)*1
	list_output_class2 = []
	for i in range(len(result_class)):
		if result_class[i] == 1:
			list_output_class2.append(class_1)
		elif result_class[i] == 0:
			list_output_class2.append(class_0)
		else:
			print("error in 'result_class' in %d data!!"%i)
	accuracy_logistic_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)

def pdf_normal_dist(value, m, s):
	return (1/(np.sqrt(2*np.pi)*s))*np.exp(-(value-m)**2/(2*(s**2)))

def classification_Naivebayes():
	global data3, data1, data1_x, data1_test, data1_test_x, accuracy_Naive_bayes_training, accuracy_Naive_bayes_testing, list_output_class1, list_output_class2, name, predicted_result1, predicted_result2, temp_nb_0, temp_nb_1, data11, data11_test
	name = "Naive_bayes"
	data1_x = data11.drop(col_num, axis=1)
	data1_y = data11.iloc[:, col_num]
	data1_test_y = data11_test.iloc[:, col_num]
	class_1 = int(max(data1_y))
	class_0 = int(min(data1_y))
	data2_y = ((data1_y==class_1)*1) #class value값이 max값(2)를 1로, 1을 0으로.
	data3 = pd.concat((data1_x, data2_y), axis=1)
	num_of_cols = len(data3.columns)

	col_continuous = []
	col_discrete = []
	for i in range(num_of_cols):
		if len(data3.iloc[:, i].unique()) > 10:
			col_continuous.append(i)
		else:
			col_discrete.append(i)
	data3_cla1 = data3[data3.iloc[:, col_num] == 1]
	data3_cla0 = data3[data3.iloc[:, col_num] == 0]
	#data3_cla1
	temp_nb_1 = []
	for i in range(num_of_cols):
		if i in col_continuous:
			temp_nb_1.append([i, np.mean(data3_cla1.iloc[:, i]), np.std(data3_cla1.iloc[:, i])])
		else:
			temp_nb_1.append([i, data3_cla1[[i]].groupby(i).size()/sum(data3_cla1[[i]].groupby(i).size())])
	#data3_cla0
	temp_nb_0 = []
	for i in range(num_of_cols):
		if i in col_continuous:
			temp_nb_0.append([i, np.mean(data3_cla0.iloc[:, i]), np.std(data3_cla0.iloc[:, i])])
		else:
			temp_nb_0.append([i, data3_cla0[[i]].groupby(i).size()/sum(data3_cla0[[i]].groupby(i).size())])
	#training
	list_output_class1 = []
	predicted_result1 =[]
	for i in range(n):
		start_value11 = 1
		start_value12 = 1
		for j in range(num_of_cols-1):
			if math.isnan(np.array(data3, dtype=float)[i, j]):
				pass
			else:
				if j in col_continuous:
					start_value11 = start_value11 * pdf_normal_dist(data3.iloc[i, j], temp_nb_1[j][1], temp_nb_1[j][2])
				else:
					start_value11 = start_value11 * temp_nb_1[j][1].loc[data3.iloc[i, j]]
		start_value11 = start_value11 * (data3_cla1.shape[0] / n)
		for j in range(num_of_cols-1):
			if math.isnan(np.array(data3, dtype=float)[i, j]):
				pass
			else:
				if j in col_continuous:
					start_value12 = start_value12 * pdf_normal_dist(data3.iloc[i, j], temp_nb_0[j][1], temp_nb_0[j][2])
				else:
					start_value12 = start_value12 * temp_nb_0[j][1].loc[data3.iloc[i, j]]
		start_value12 = start_value12 * (data3_cla0.shape[0] / n)

		prob1 = start_value11 / (start_value11+start_value12)
		prob2 = start_value12 / (start_value11+start_value12)
		temp_output = []
		temp_output.append(prob1)
		temp_output.append(prob2)
		predicted_result1.append(prob1)

		if temp_output[0]>temp_output[1]:
			list_output_class1.append(class_1)
		else:
			list_output_class1.append(class_0)
	data1 = np.array(data11)
	accuracy_Naive_bayes_training = sum(data1_y == list_output_class1)/len(data1_y)
	#testing
	list_output_class2 = []
	predicted_result2 =[]
	for i in range(data11_test.shape[0]):
		start_value21 = 1
		start_value22 = 1
		for j in range(num_of_cols-1):
			if math.isnan(np.array(data11_test, dtype=float)[i, j]):
				pass
			else:
				if j in col_continuous:
					start_value21 = start_value21 * pdf_normal_dist(data11_test.iloc[i, j], temp_nb_1[j][1], temp_nb_1[j][2])
				else:
					start_value21 = start_value21 * temp_nb_1[j][1].loc[data11_test.iloc[i, j]]
		start_value21 = start_value21 * (data3_cla1.shape[0] / n)
		for j in range(num_of_cols-1):
			if math.isnan(np.array(data11_test, dtype=float)[i, j]):
				pass
			else:
				if j in col_continuous:
					start_value22 = start_value22 * pdf_normal_dist(data11_test.iloc[i, j], temp_nb_0[j][1], temp_nb_0[j][2])
				else:
					start_value22 = start_value22 * temp_nb_0[j][1].loc[data11_test.iloc[i, j]]
		start_value22 = start_value22 * (data3_cla0.shape[0] / n)

		prob1 = start_value21 / (start_value21+start_value22)
		prob2 = start_value22 / (start_value21+start_value22)
		temp_output = []
		temp_output.append(prob1)
		temp_output.append(prob2)
		predicted_result2.append(prob1)

		if temp_output[0]>temp_output[1]:
			list_output_class2.append(class_1)
		else:
			list_output_class2.append(class_0)
	data1_test = np.array(data11_test)
	accuracy_Naive_bayes_testing = sum(data1_test_y == list_output_class2)/len(data1_y)

def impurity(input_data_class):
	input_data_class_unique = np.unique(input_data_class)
	temp7 = []
	for i in range(len(input_data_class_unique)):
		temp7.append(sum(input_data_class==input_data_class_unique[i])/len(input_data_class))
	temp8 = np.array(temp7)**2
	temp9 = sum(temp8)
	imp_t = 1-temp9
	return imp_t

def classification_one_level_decision_tree():
	global data1, data1_x, data1_test, data1_test_x, data1_y, data1_test_y, accuracy_1_level_decision_tree_training, accuracy_1_level_decision_tree_testing, list_output_class1, list_output_class2, name, axis_name, boundary, left_side, right_side, class_1, class_0
	name = "1-level_decision_tree"
	data1_x = data11.drop(col_num, axis=1)
	data1_y = data11.iloc[:, col_num]
	data1_test_x = data11_test.drop(col_num, axis=1)
	data1_test_y = data11_test.iloc[:, col_num]
	class_1 = int(max(data1_y))
	class_0 = int(min(data1_y))
	data2_y = ((data1_y==class_1)*1) #class value값이 max값(2)를 1로, 1을 0으로.
	data3 = pd.concat((data1_x, data2_y), axis=1)
	num_of_cols = len(data3.columns)

	#boundary, goodness_of_split
	boundary_gof_1 = []
	for j in range(p-1):
		input_x = np.array(data1_x.iloc[:, j]).reshape(n, 1)
		input_y = np.array(data1_y).reshape(n, 1)
		input_data = np.concatenate((input_x, input_y), axis=1)
		input_data2 = input_data[np.argsort(input_data[:, 0])] #sorted
		input_data_x_val = input_data2[:, 0]
		input_data_x_cla = input_data2[:, 1]
		input_data_x_unique = np.unique(input_data_x_val)

		#impurity_total(j_th col)
		imp_t = impurity(input_data_x_cla)
		#impurity_by_boundary
		boundary_list = []
		for i in range(len(input_data_x_unique)-1):
			boundary_list.append((input_data_x_unique[i]+input_data_x_unique[i+1])/2)

		imp_list2 = []#impurity_list
		for k in range(len(boundary_list)):
			boundary = boundary_list[k]
			left_side = input_data2[input_data_x_val < boundary]
			right_side = input_data2[input_data_x_val >= boundary]
			goodness_of_split = imp_t-impurity(left_side[:, 1])*(len(left_side)/(len(left_side)+len(right_side)))-impurity(right_side[:, 1])*(len(right_side)/(len(left_side)+len(right_side)))
			imp_list = []
			imp_list.append(boundary_list[k])
			imp_list.append(goodness_of_split)
			imp_list2.append(imp_list)

		imp_list3 = np.array(imp_list2)
		output1 = imp_list3[np.argmax(imp_list3[:, 1]), :]
		boundary_gof_1.append(output1)

	#final_output (specific boundary)
	j = np.argmax(np.array(boundary_gof_1)[:, 1])
	axis_name = "x"+str(j+1)
	input_x = np.array(data1_x.iloc[:, j])
	input_x = input_x.reshape(n, 1)
	input_data = np.concatenate((input_x, input_y), axis=1)
	input_data2 = input_data[np.argsort(input_data[:, 0])] #sorted
	input_data_x_val = input_data2[:, 0]
	input_data_x_cla = input_data2[:, 1]
	input_data_x_unique = np.unique(input_data_x_val)

	boundary = boundary_gof_1[np.argmax(np.array(boundary_gof_1)[:, 1])][0]
	left_side = input_data2[input_data_x_val < boundary]
	right_side = input_data2[input_data_x_val >= boundary]
	goodness_of_split = imp_t-impurity(left_side[:, 1])*(len(left_side)/(len(left_side)+len(right_side)))-impurity(right_side[:, 1])*(len(right_side)/(len(left_side)+len(right_side)))

	list_output_class1 = []
	for i in range(len(data1)):
		if data1.iloc[i, j]<boundary:
			list_output_class1.append(class_1)
		else:
			list_output_class1.append(class_0)

	data1 = np.array(data1)
	accuracy_1_level_decision_tree_training = sum(data1[:, p-1] == list_output_class1)/len(data1)

	#testing
	n2 = len(data1_test_x)
	test_input_x = np.array(data1_test_x.iloc[:, j]).reshape(n2, 1)
	test_input_y = np.array(data1_test_y).reshape(n2, 1)
	test_input_data = np.concatenate((test_input_x, test_input_y), axis=1)
	test_input_data2 = test_input_data[np.argsort(test_input_data[:, 0])] #sorted
	test_input_data_x_val = test_input_data2[:, 0]
	test_input_data_x_cla = test_input_data2[:, 1]
	test_input_data_x_unique = np.unique(test_input_data_x_val)

	list_output_class2 = []
	for i in range(len(data1_test)):
		if data1_test.iloc[i, j]<boundary:
			list_output_class2.append(class_1)
		else:
			list_output_class2.append(class_0)

	data1_test = np.array(data1_test)
	accuracy_1_level_decision_tree_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)

def combinations(n, input_list, output_list=[]):
    if output_list is None:
        output_list = []
    if len(input_list) == n:
        if output_list.count(input_list) == 0:
            output_list.append(input_list)
            output_list.sort()
        return output_list
    else:
        for i in range(len(input_list)):
            refined_list = input_list[:i] + input_list[i+1:]
            output_list = combinations(n, refined_list, output_list)
        return output_list

def classification_one_level_decision_tree_2():
	global data1, data1_x, data1_test, data1_test_x, data1_y, data1_test_y, accuracy_1_level_decision_tree_training, accuracy_1_level_decision_tree_testing, list_output_class1, list_output_class2, name, axis_name, boundary, left_side, right_side, class_1, class_0
	name = "1-level_decision_tree"

	data1_x = data1.drop(columns = col_name)
	data1_y = data1[[col_name]]
	data1_test_x = data1_test.drop(columns = col_name)
	data1_test_y = data1_test[[col_name]]

	#preprocessing
	data1_y = data1_y.replace(list_classes[0], 0) #'No' -> 0
	data1_y = data1_y.replace(list_classes[1], 1) #'Yes' -> 1
	data1_y = data1_y.apply(pd.to_numeric)

	class_1 = int(max(np.array(data1_y))) #class_1 == 1
	class_0 = int(min(np.array(data1_y))) #class_0 == 0

	data2_y = np.array(data1_y) #np.arraay
	data3 = pd.concat((data1_x, data1_y), axis=1)
	num_of_cols = len(data3.columns)

	boundary_gof_1 = []
	for i in range(p-1): #goodness of split by one specific column
		input_x = np.array(data1_x.iloc[:, i]).reshape(n, 1)
		input_y = np.array(data1_y).reshape(n, 1)
		input_data = np.concatenate((input_x, input_y), axis=1)
		input_data2 = input_data[np.argsort(input_data[:, 0])] #sorted
		input_data_x_val = input_data2[:, 0]
		input_data_x_unique = np.unique(input_data_x_val)
		input_data_x_cla = input_data2[:, 1]
		#impurity_total
		imp_t = impurity(input_data_x_cla)
		#impurity_by_boundary
		boundary_list = []
		temp_list = list(range(len(input_data_x_unique)))
		temp_set = set(temp_list)
		temp=[]
		for j in range(1, len(temp_list)):
			result = combinations(n=j, input_list=temp_list, output_list=[])
			for j in range(len(result)):
				a = sorted([result[j], list(temp_set.difference(set(result[j])))])
				if a in temp:
					pass
				else:
					temp.append(a)
		imp_list2 = []
		dim = np.array(temp).shape
		for j in range(dim[0]):
			b_left = []
			b_right = []
			for k in range(2):
				for l in range(len(temp[j][k])):
					b = input_data_x_unique[temp[j][k][l]]
					if k == 0:
						b_left.append(b)
					elif k == 1:
						b_right.append(b)
					else:
						pass
			left_side = np.zeros(shape=(1,2))
			right_side = np.zeros(shape=(1,2))
			cnt1 = 0
			cnt2 = 0
			for k in range(len(b_left)):
				temp1 = input_data2[input_data2[:, 0] == b_left[k]]
				left_side = np.concatenate((left_side, temp1), axis=0)
				cnt1 += sum(input_data2[:, 0] == b_left[k])
			for k in range(len(b_right)):
				temp2 = input_data2[input_data2[:, 0] == b_right[k]]
				right_side = np.concatenate((right_side, temp2), axis=0)
				cnt2 += sum(input_data2[:, 0] == b_right[k])
			left_side = left_side[1:, :]
			right_side = right_side[1:, :]

			goodness_of_split = imp_t-impurity(left_side[:, 1])*(len(left_side)/(len(left_side)+len(right_side)))-impurity(right_side[:, 1])*(len(right_side)/(len(left_side)+len(right_side)))

			imp_list = [[b_left, b_right], goodness_of_split]
			imp_list2.append(imp_list)

		imp_list3 = np.array(imp_list2)
		output1 = imp_list3[np.argmax(imp_list3[:, 1]), :]
		boundary_gof_1.append(output1)

	#final_output (specific boundary)
	h = np.argmax(np.array(boundary_gof_1)[:, 1])
	axis_name = data1.columns[h]
	input_x = np.array(data1_x.iloc[:, h])
	input_x = input_x.reshape(n, 1)
	input_data = np.concatenate((input_x, input_y), axis=1)
	input_data2 = input_data[np.argsort(input_data[:, 0])] #sorted
	input_data_x_val = input_data2[:, 0]
	input_data_x_cla = input_data2[:, 1]
	input_data_x_unique = np.unique(input_data_x_val)

	imp_t = impurity(input_data_x_cla)

	temp_list = list(range(len(input_data_x_unique)))
	temp_set = set(temp_list)

	temp=[]
	for i in range(1, len(temp_list)):
		result = combinations(n=i, input_list=temp_list, output_list=[])
		for j in range(len(result)):
			a = sorted([result[j], list(temp_set.difference(set(result[j])))])
			if a in temp:
				pass
			else:
				temp.append(a)

	imp_list2 = []
	dim = np.array(temp).shape

	boundary = boundary_gof_1[np.argmax(np.array(boundary_gof_1)[:, 1])][0]
	b_left = []
	b_right = []

	for i in range(2):
		for j in range(len(temp[0][i])):
			b = input_data_x_unique[temp[0][i][j]]
			if i == 0:
				b_left.append(b)
			elif i == 1:
				b_right.append(b)
			else:
				pass

	left_side = np.zeros(shape=(1,2))
	right_side = np.zeros(shape=(1,2))
	cnt1 = 0
	cnt2 = 0
	for i in range(len(b_left)):
		temp1 = input_data2[input_data2[:, 0] == b_left[i]]
		left_side = np.concatenate((left_side, temp1), axis=0)
		cnt1 += sum(input_data2[:, 0] == b_left[i])
	for i in range(len(b_right)):
		temp2 = input_data2[input_data2[:, 0] == b_right[i]]
		right_side = np.concatenate((right_side, temp2), axis=0)
		cnt2 += sum(input_data2[:, 0] == b_right[i])
	left_side = left_side[1:, :]
	right_side = right_side[1:, :]

	goodness_of_split = imp_t-impurity(left_side[:, 1])*(len(left_side)/(len(left_side)+len(right_side)))-impurity(right_side[:, 1])*(len(right_side)/(len(left_side)+len(right_side)))

	list_output_class1 = []
	for i in range(len(data1)):
		for j in range(len(b_left)):
			if data1.iloc[i, h] == b_left[j]:
				list_output_class1.append(list_classes[1])
			else:
				list_output_class1.append(list_classes[0])

	data1 = np.array(data1)
	accuracy_1_level_decision_tree_training = sum(data1[:, p-1] == list_output_class1)/len(data1)

	#testing
	n2 = len(data1_test_x)
	test_input_x = np.array(data1_test_x.iloc[:, j]).reshape(n2, 1)
	test_input_y = np.array(data1_test_y).reshape(n2, 1)
	test_input_data = np.concatenate((test_input_x, test_input_y), axis=1)
	test_input_data2 = test_input_data[np.argsort(test_input_data[:, 0])] #sorted
	test_input_data_x_val = test_input_data2[:, 0]
	test_input_data_x_cla = test_input_data2[:, 1]
	test_input_data_x_unique = np.unique(test_input_data_x_val)

	list_output_class2 = []
	for i in range(len(data1_test)):
		for j in range(len(b_left)):
			if data1_test.iloc[i, h] == b_left[j]:
				list_output_class2.append(list_classes[1])
			else:
				list_output_class2.append(list_classes[0])

	data1_test = np.array(data1_test)
	accuracy_1_level_decision_tree_testing = sum(data1_test[:, p-1] == list_output_class2)/len(data1_test)

def classification():
	global what, num_of_class, continuous_or_categorical
	num_of_class = len(set(np.array(data1.iloc[:, col_num])))
	if num_of_class>2:
		print("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')")
		what = int(input("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')"))
	else:
		print("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA' or 4 = 'Logistic' or 5 = 'Naive_bayes' or 6 = '1-level_decision_tree')")
		what = int(input("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA' or 4 = 'Logistic' or 5 = 'Naive_bayes' or 6 = '1-level_decision_tree')"))

	if what == 1:
		classification_LDA()
	elif what == 2:
		classification_QDA()
	elif what == 3:
		classification_RDA()
	elif what == 4:
		classification_logistic()
	elif what == 5:
		classification_Naivebayes()
	elif what == 6:
		print("Select the data type(1 = 'continuous data' or 2= 'categorical data')")
		continuous_or_categorical = input("Select the data type(1 = 'continuous data' or 2= 'categorical data')")
		if continuous_or_categorical == 1:
			classification_one_level_decision_tree()
		else:
			classification_one_level_decision_tree_2()
	else:
		print('error!!! Select the classifier again!!!')

def get_confusion_matrix():
	global df_confusion_matrix_training, df_confusion_matrix_testing, confusion_matrix_training, confusion_matrix_testing, sensitivity_logistic_training, sensitivity_logistic_testing, specificity_logistic_training, specificity_logistic_testing,  sensitivity_Naive_bayes_training, sensitivity_Naive_bayes_testing, specificity_Naive_bayes_training, specificity_Naive_bayes_testing, sensitivity_1_level_decision_tree_training, specificity_1_level_decision_tree_training, sensitivity_1_level_decision_tree_testing, specificity_1_level_decision_tree_testing
	if what == 6:
		if continuous_or_categorical == 1:
			confusion_matrix_training= []
			for k in range(len(list_classes)):
				i = k+1
				for l in range(len(list_classes)):
					j = l+1
					confusion_matrix_training.append(sum((data1[:, p-1]==i) & (np.array(list_output_class1)==j)))
			df_confusion_matrix_training = pd.DataFrame(np.array(confusion_matrix_training).reshape(K,K))
			df_confusion_matrix_training.index = np.arange(1, len(df_confusion_matrix_training) + 1)
			df_confusion_matrix_training.columns = np.arange(1, len(df_confusion_matrix_training) +1)

			confusion_matrix_testing= []
			for k in range(len(list_classes)):
				i = k+1
				for l in range(len(list_classes)):
					j = l+1
					confusion_matrix_testing.append(sum((data1_test[:, p-1]==i) & (np.array(list_output_class2)==j)))
			df_confusion_matrix_testing = pd.DataFrame(np.array(confusion_matrix_testing).reshape(K,K))
			df_confusion_matrix_testing.index = np.arange(1, len(df_confusion_matrix_testing) + 1)
			df_confusion_matrix_testing.columns = np.arange(1, len(df_confusion_matrix_testing) +1)
		else:
			confusion_matrix_training= []
			for i in list_classes:
				for j in list_classes:
					confusion_matrix_training.append(sum((data1[:, p-1]==i) & (np.array(list_output_class1)==j)))
			df_confusion_matrix_training = pd.DataFrame(np.array(confusion_matrix_training).reshape(K,K))
			df_confusion_matrix_training.index = np.arange(1, len(df_confusion_matrix_training) + 1)
			df_confusion_matrix_training.columns = np.arange(1, len(df_confusion_matrix_training) +1)

			confusion_matrix_testing= []
			for i in list_classes:
				for j in list_classes:
					confusion_matrix_testing.append(sum((data1_test[:, p-1]==i) & (np.array(list_output_class2)==j)))
			df_confusion_matrix_testing = pd.DataFrame(np.array(confusion_matrix_testing).reshape(K,K))
			df_confusion_matrix_testing.index = np.arange(1, len(df_confusion_matrix_testing) + 1)
			df_confusion_matrix_testing.columns = np.arange(1, len(df_confusion_matrix_testing) +1)
	else:
		confusion_matrix_training= []
		for k in range(len(list_classes)):
			i = k+1
			for l in range(len(list_classes)):
				j = l+1
				confusion_matrix_training.append(sum((data1[:, p-1]==i) & (np.array(list_output_class1)==j)))
		df_confusion_matrix_training = pd.DataFrame(np.array(confusion_matrix_training).reshape(K,K))
		df_confusion_matrix_training.index = np.arange(1, len(df_confusion_matrix_training) + 1)
		df_confusion_matrix_training.columns = np.arange(1, len(df_confusion_matrix_training) +1)

		confusion_matrix_testing= []
		for k in range(len(list_classes)):
			i = k+1
			for l in range(len(list_classes)):
				j = l+1
				confusion_matrix_testing.append(sum((data1_test[:, p-1]==i) & (np.array(list_output_class2)==j)))
		df_confusion_matrix_testing = pd.DataFrame(np.array(confusion_matrix_testing).reshape(K,K))
		df_confusion_matrix_testing.index = np.arange(1, len(df_confusion_matrix_testing) + 1)
		df_confusion_matrix_testing.columns = np.arange(1, len(df_confusion_matrix_testing) +1)

	if what == 4:
		sensitivity_logistic_training = df_confusion_matrix_training.iloc[1, 1]/(df_confusion_matrix_training.iloc[1, 0]+df_confusion_matrix_training.iloc[1, 1])
		specificity_logistic_training = df_confusion_matrix_training.iloc[0, 0]/(df_confusion_matrix_training.iloc[0, 0]+df_confusion_matrix_training.iloc[0, 1])
		sensitivity_logistic_testing = df_confusion_matrix_testing.iloc[1, 1]/(df_confusion_matrix_testing.iloc[1, 0]+df_confusion_matrix_testing.iloc[1, 1])
		specificity_logistic_testing = df_confusion_matrix_testing.iloc[0, 0]/(df_confusion_matrix_testing.iloc[0, 0]+df_confusion_matrix_testing.iloc[0, 1])
	elif what == 5:
		sensitivity_Naive_bayes_training = df_confusion_matrix_training.iloc[1, 1]/(df_confusion_matrix_training.iloc[1, 0]+df_confusion_matrix_training.iloc[1, 1])
		specificity_Naive_bayes_training = df_confusion_matrix_training.iloc[0, 0]/(df_confusion_matrix_training.iloc[0, 0]+df_confusion_matrix_training.iloc[0, 1])
		sensitivity_Naive_bayes_testing = df_confusion_matrix_testing.iloc[1, 1]/(df_confusion_matrix_testing.iloc[1, 0]+df_confusion_matrix_testing.iloc[1, 1])
		specificity_Naive_bayes_testing = df_confusion_matrix_testing.iloc[0, 0]/(df_confusion_matrix_testing.iloc[0, 0]+df_confusion_matrix_testing.iloc[0, 1])
	elif what == 6:
		sensitivity_1_level_decision_tree_training = df_confusion_matrix_training.iloc[1, 1]/(df_confusion_matrix_training.iloc[1, 0]+df_confusion_matrix_training.iloc[1, 1])
		specificity_1_level_decision_tree_training = df_confusion_matrix_training.iloc[0, 0]/(df_confusion_matrix_training.iloc[0, 0]+df_confusion_matrix_training.iloc[0, 1])
		sensitivity_1_level_decision_tree_testing = df_confusion_matrix_testing.iloc[1, 1]/(df_confusion_matrix_testing.iloc[1, 0]+df_confusion_matrix_testing.iloc[1, 1])
		specificity_1_level_decision_tree_testing = df_confusion_matrix_testing.iloc[0, 0]/(df_confusion_matrix_testing.iloc[0, 0]+df_confusion_matrix_testing.iloc[0, 1])
	else:
		pass

def save_as_file_classification():
	with open("HW8KangJH_python_"+name+"_output.txt","w") as txt:
		if what==3:
			print("alpha = ", alpha, "\ngamma = ", gamma, "\n", sep="", file=txt)
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(list_RDA_2[:, 0], list_RDA_2[:, 1], list_RDA_2[:, 2], c='r')
			ax.set_xlabel('alpha')
			ax.set_ylabel('gamma')
			ax.set_zlabel('accuracy_RDA_testing')
			plt.show()
		elif what==6:
			if continuous_or_categorical == 1:
				print("Tree Structure", sep="", file=txt)
				print("      Node 1: %s <= %f (%d, %d)"%(axis_name, boundary, (sum(left_side[:, 1] == max(data1_y))+sum(right_side[:, 1] == max(data1_y))), (sum(left_side[:, 1] == min(data1_y))+sum(right_side[:, 1] == min(data1_y)))), sep="", file=txt)
				print("        Node 2: %d (%d, %d)"%(class_1, sum(left_side[:, 1] == max(data1_y)), sum(left_side[:, 1] == min(data1_y))), sep="", file=txt)
				print("        Node 3: %d (%d, %d)\n"%(class_0, sum(right_side[:, 1] == max(data1_y)), sum(right_side[:, 1] == min(data1_y))), sep="", file=txt)
			else:
				print("Tree Structure", sep="", file=txt)
				print("      Node 1: %s in %s (%d, %d)"%(axis_name, set(boundary[0]), (sum(left_side[:, 1] == max(data1_y.values)[0])+sum(right_side[:, 1] == max(data1_y.values)[0])), (sum(left_side[:, 1] == min(data1_y.values)[0])+sum(right_side[:, 1] == min(data1_y.values)[0]))), sep="", file=txt)
				print("        Node 2: %s (%d, %d)"%(list_classes[1], sum(left_side[:, 1] == max(data1_y.values)[0]), sum(left_side[:, 1] == min(data1_y.values)[0])), sep="", file=txt)
				print("        Node 3: %s (%d, %d)\n"%(list_classes[0], sum(right_side[:, 1] == max(data1_y.values)[0]), sum(right_side[:, 1] == min(data1_y.values)[0])), sep="", file=txt)
		if what==4:
			print("ID, Actual class, Resub pred, Pred Prob", "\n", "-----------------------------", sep="", file=txt)
			for k in range(len(data1)):
				print(k+1, ", ", int(data1[k, col_num]), ", ", list_output_class1[k],  ", ", round(predicted_result1[k][0], 3), sep="", file=txt)
		elif what==5:
			print("ID, Actual class, Resub pred, Pred Prob", "\n", "-----------------------------", sep="", file=txt)
			for k in range(len(data1)):
				print(k+1, ", ", int(data1[k, col_num]), ", ", list_output_class1[k],  ", ", round(predicted_result1[k], 3), sep="", file=txt)
		else:
			if continuous_or_categorical == 1:
				print("ID, Actual class, Resub pred", "\n", "-----------------------------", sep="", file=txt)
				for k in range(len(data1)):
					print(k+1, ", ", int(data1[k, col_num]), ", ", list_output_class1[k], sep="", file=txt)
			else:
				print("ID, Actual class, Resub pred", "\n", "-----------------------------", sep="", file=txt)
				for k in range(len(data1)):
					print(k+1, ", ", data1[k, col_num], ", ", list_output_class1[k], sep="", file=txt)
		print("", file=txt)
		print("Confusion Matrix (Resubstitution)", "\n", "----------------------------------", file=txt)
		print(df_confusion_matrix_training, file=txt)
		print("", file=txt)
		print("Model Summary (Resubstitution)", "\n", "------------------------------", sep="", file=txt)
		if what==1:
			print("Overall accuracy = .", str(round(accuracy_LDA_training, 3))[2:], sep="", file=txt)
		elif what==2:
			print("Overall accuracy = .", str(round(accuracy_QDA_training, 3))[2:], sep="", file=txt)
		elif what==3:
			print("Overall accuracy = .", str(round(accuracy_RDA_training, 3))[2:], sep="", file=txt)
		elif what==4:
			print("Overall accuracy = .", str(round(accuracy_logistic_training, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_logistic_training, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_logistic_training, 3))[2:], sep="", file=txt)
		elif what==5:
			print("Overall accuracy = .", str(round(accuracy_Naive_bayes_training, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_Naive_bayes_training, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_Naive_bayes_training, 3))[2:], sep="", file=txt)
		elif what==6:
			print("Overall accuracy = .", str(round(accuracy_1_level_decision_tree_training, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_1_level_decision_tree_training, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_1_level_decision_tree_training, 3))[2:], sep="", file=txt)
		else:
			print("error!", file=txt)
		print("\n", file=txt)
		if what==4:
			print("ID, Actual class, Test pred, Pred Prob", "\n", "-----------------------------", sep="", file=txt)
			for k in range(len(data1_test)):
				print(k+1, ", ", int(data1_test[k, col_num]), ", ", list_output_class2[k],  ", ", round(predicted_result2[k][0], 3), sep="", file=txt)
		elif what==5:
			print("ID, Actual class, Test pred, Pred Prob", "\n", "-----------------------------", sep="", file=txt)
			for k in range(len(data1_test)):
				print(k+1, ", ", int(data1_test[k, col_num]), ", ", list_output_class2[k],  ", ", round(predicted_result2[k], 3), sep="", file=txt)
		else:
			if continuous_or_categorical == 1:
				print("ID, Actual class, Test pred", "\n", "-----------------------------", sep="", file=txt)
				for k in range(len(data1_test)):
					print(k+1, ", ", int(data1_test[k, col_num]), ", ", list_output_class2[k], sep="", file=txt)
			else:
				print("ID, Actual class, Test pred", "\n", "-----------------------------", sep="", file=txt)
				for k in range(len(data1_test)):
					print(k+1, ", ", data1_test[k, col_num], ", ", list_output_class2[k], sep="", file=txt)
		print("", file=txt)
		print("Confusion Matrix (Test)", "\n", "----------------------------------", file=txt)
		print(df_confusion_matrix_testing, file=txt)
		print("", file=txt)
		print("Model Summary (Test)", "\n", "------------------------------", sep="", file=txt)
		if what==1:
			print("Overall accuracy = .", str(round(accuracy_LDA_testing, 3))[2:], sep="", file=txt)
		elif what==2:
			print("Overall accuracy = .", str(round(accuracy_QDA_testing, 3))[2:], sep="", file=txt)
		elif what==3:
			print("Overall accuracy = .", str(round(accuracy_RDA_testing, 3))[2:], sep="", file=txt)
		elif what==4:
			print("Overall accuracy = .", str(round(accuracy_logistic_testing, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_logistic_testing, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_logistic_testing, 3))[2:], sep="", file=txt)
		elif what==5:
			print("Overall accuracy = .", str(round(accuracy_Naive_bayes_testing, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_Naive_bayes_testing, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_Naive_bayes_testing, 3))[2:], sep="", file=txt)
		elif what==6:
			print("Overall accuracy = .", str(round(accuracy_1_level_decision_tree_testing, 3))[2:], sep="", file=txt)
			print("Sensitivity = .", str(round(sensitivity_1_level_decision_tree_testing, 3))[2:], sep="", file=txt)
			print("Specificity = .", str(round(specificity_1_level_decision_tree_testing, 3))[2:], sep="", file=txt)
		else:
			print("error!", file=txt)

################################
#2. Run
################################

reg_or_cla()
