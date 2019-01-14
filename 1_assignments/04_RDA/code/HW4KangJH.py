import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

################################
#1. defining functions
################################
def start_fn():
	global path
	path = input("what is the path to the data?")#C:/Users/YISS/Desktop/data_mining/hw/hw4
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
	with open("HW4KangJH_python_regression_output.txt","w") as txt:
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
#1-2. classification(LDA & QDA)
################################
def training_data():
	global data_name
	print("Enter the training data file name: ")
	data_name = input("Enter the training data file name: ")

def testing_data():
	global data_name2
	print("Enter the testing data file name: ")
	data_name2 = input("Enter the testing data file name: ")

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
	global n, p, list_classes, list_classes_cnt, K, list_p, list_S, Sp, Sp_inv
	n = data1.shape[0]
	p = data1.shape[1]
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
		data1_k = data1[data1[col_num]==k]
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

def classification_LDA():
	global data1, data1_x, data1_test, data1_test_x, accuracy_LDA_training, accuracy_LDA_testing, list_output_class1, list_output_class2
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
	global data1, data1_x, data1_test, data1_test_x, accuracy_QDA_training, accuracy_QDA_testing, list_output_class1, list_output_class2
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
	global data1, data1_x, data1_test, data1_test_x, accuracy_RDA_training, accuracy_RDA_testing, list_output_class1, list_output_class2, alpha, gamma, list_RDA_2
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

def classification():
	global what, LQR
	print("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')")
	what = int(input("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')"))
	if what == 1:
		LQR = 'output_LDA'
		classification_LDA()
	elif what == 2:
		LQR = 'output_QDA'
		classification_QDA()
	elif what == 3:
		LQR = 'output_RDA'
		classification_RDA()
	else:
		print('error!!! Select the classifier again!!!')

def get_confusion_matrix():
	global df_confusion_matrix_training, df_confusion_matrix_testing, confusion_matrix_training, confusion_matrix_testing
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

def save_as_file_classification():
	with open("HW4KangJH_python_classification_output.txt","w") as txt:
		if what==3:
			print("alpha = ", alpha, "\ngamma = ", gamma, "\n", sep="", file=txt)
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(list_RDA_2[:, 0], list_RDA_2[:, 1], list_RDA_2[:, 2], c='r')
			ax.set_xlabel('alpha')
			ax.set_ylabel('gamma')
			ax.set_zlabel('accuracy_RDA_testing')
			plt.show()
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
		else:
			print("error!", file=txt)
		print("\n", file=txt)
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
		else:
			print("error!", file=txt)

################################
#2. Run
################################

reg_or_cla()
