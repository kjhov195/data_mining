import os
import numpy as np
import pandas as pd
from sklearn import tree

def start_fn():
	global path
	path = input("what is the path to the data?")
	os.chdir(path)

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

def categorical_col():
	global col_num2
	print("Select the number of the categorical variables column(ex. 1, 2, 3, ...)")
	col_num2 = int(input("Select the number of the categorical variables column(ex. 1, 2, 3, ...)"))
	col_num2 -= 1

def classification_decision_tree():
	global name, accuracy2, accuracy22, list_output_class2, list_output_class22, data2_test, df_confusion_matrix_testing, df_confusion_matrix_testing2
	#decision Tree
	name = 'decision-tree'

	#depth 2
	depth = 2

	data2 = np.array(data1).copy()
	data_x = np.delete(data2.copy(), col_num, 1)
	data_y = data2[:, col_num]
	data2_test = np.array(data1_test).copy()
	data_x_test = np.delete(data2_test.copy(), col_num, 1)
	data_y_test = data2_test[:, col_num]

	#training
	#bagging
	n = data2.shape[0]
	list_accuracy1 = []
	for i in range(51):
		np.random.seed(i)
		list_idx = np.arange(0, n)
		selected_idx = np.random.choice(list_idx, size=list_idx.shape, replace=True)
		data3 = data2.copy()[selected_idx, :]
		data_x = np.delete(data3.copy(), col_num, 1)
		data_y = data3.copy()[:, col_num]

		clf = tree.DecisionTreeClassifier(max_depth=depth)
		clf = clf.fit(data_x, data_y)

		list_output_class1 = []
		for j in range(len(data_x)):
			temp1 = clf.predict([data_x[j, :]])
			list_output_class1.append(temp1[0])

		accuracy1 = sum(list_output_class1 == data_y)/n
		list_accuracy1.append(accuracy1)

	idx_argmax = np.argmax(list_accuracy1)

	#tree
	np.random.seed(idx_argmax)
	list_idx = np.arange(0, n)
	selected_idx = np.random.choice(list_idx, size=list_idx.shape, replace=True)
	data3 = data2.copy()[selected_idx, :]
	data_x = np.delete(data3.copy(), col_num, 1)
	data_y = data3.copy()[:, col_num]

	clf = tree.DecisionTreeClassifier(max_depth=depth)
	clf = clf.fit(data_x, data_y)

	list_output_class1 = []
	for i in range(len(data_x)):
		temp1 = clf.predict([data_x[i, :]])
		list_output_class1.append(temp1[0])

	accuracy1 = sum(list_output_class1 == data_y)/n

	#testing
	list_output_class2 = []
	n2 = data2_test.shape[0]
	for i in range(n2):
		temp_prob = clf.predict_proba([data_x_test[i, :]])
		temp_class = clf.predict([data_x_test[i, :]])[0]
		list_output_class2.append(temp_class)

	accuracy2 = sum(list_output_class2 == data_y_test)/n2

	#confusion matrx
	list_classes = np.array(np.unique(data_y), dtype='int')
	K = len(list_classes)

	confusion_matrix_testing = []
	for k in range(len(list_classes)):
		i = k+1
		for l in range(len(list_classes)):
			j = l+1
			confusion_matrix_testing.append(sum((data2_test[:, col_num]==i) & (np.array(list_output_class2)==j)))
	df_confusion_matrix_testing = pd.DataFrame(np.array(confusion_matrix_testing).reshape(K,K))
	df_confusion_matrix_testing.index = np.arange(1, len(df_confusion_matrix_testing) + 1)
	df_confusion_matrix_testing.columns = np.arange(1, len(df_confusion_matrix_testing) +1)

	#depth 4
	depth = 4

	data2 = np.array(data1).copy()
	data_x = np.delete(data2.copy(), col_num, 1)
	data_y = data2[:, col_num]
	data2_test = np.array(data1_test).copy()
	data_x_test = np.delete(data2_test.copy(), col_num, 1)
	data_y_test = data2_test[:, col_num]

	#training
	#bagging
	n = data2.shape[0]
	list_accuracy1 = []
	for i in range(51):
		np.random.seed(i)
		list_idx = np.arange(0, n)
		selected_idx = np.random.choice(list_idx, size=list_idx.shape, replace=True)
		data3 = data2.copy()[selected_idx, :]
		data_x = np.delete(data3.copy(), col_num, 1)
		data_y = data3.copy()[:, col_num]

		clf = tree.DecisionTreeClassifier(max_depth=depth)
		clf = clf.fit(data_x, data_y)

		list_output_class1 = []
		for j in range(len(data_x)):
			temp1 = clf.predict([data_x[j, :]])
			list_output_class1.append(temp1[0])

		accuracy1 = sum(list_output_class1 == data_y)/n
		list_accuracy1.append(accuracy1)

	idx_argmax = np.argmax(list_accuracy1)

	#tree
	np.random.seed(idx_argmax)
	list_idx = np.arange(0, n)
	selected_idx = np.random.choice(list_idx, size=list_idx.shape, replace=True)
	data3 = data2.copy()[selected_idx, :]
	data_x = np.delete(data3.copy(), col_num, 1)
	data_y = data3.copy()[:, col_num]

	clf = tree.DecisionTreeClassifier(max_depth=depth)
	clf = clf.fit(data_x, data_y)

	list_output_class1 = []
	for i in range(len(data_x)):
		temp1 = clf.predict([data_x[i, :]])
		list_output_class1.append(temp1[0])

	accuracy1 = sum(list_output_class1 == data_y)/n

	#testing
	list_output_class22 = []
	n2 = data2_test.shape[0]
	for i in range(n2):
		temp_prob = clf.predict_proba([data_x_test[i, :]])
		temp_class = clf.predict([data_x_test[i, :]])[0]
		list_output_class22.append(temp_class)

	accuracy22 = sum(list_output_class22 == data_y_test)/n2

	#confusion matrx
	list_classes = np.array(np.unique(data_y), dtype='int')
	K = len(list_classes)

	confusion_matrix_testing2 = []
	for k in range(len(list_classes)):
		i = k+1
		for l in range(len(list_classes)):
			j = l+1
			confusion_matrix_testing2.append(sum((data2_test[:, col_num]==i) & (np.array(list_output_class22)==j)))
	df_confusion_matrix_testing2 = pd.DataFrame(np.array(confusion_matrix_testing2).reshape(K,K))
	df_confusion_matrix_testing2.index = np.arange(1, len(df_confusion_matrix_testing2) + 1)
	df_confusion_matrix_testing2.columns = np.arange(1, len(df_confusion_matrix_testing2) +1)

#print
def save_as_file_classification():
	with open("HW11KangJH_python_"+name+"_output.txt","w") as txt:
		print("   (1)    Tree with depth 2", sep="", file=txt)
		print("ID, Actual class, tree-depth2 pred", "\n", "-----------------------------", sep="", file=txt)
		for k in range(len(data1_test)):
			print(k+1, ", ", int(data2_test[k, col_num]), ", ", np.array(list_output_class2, dtype='int')[k], sep="", file=txt)
		print("", file=txt)
		print("Confusion Matrix (tree-depth2)", "\n", "----------------------------------", file=txt)
		print(df_confusion_matrix_testing, file=txt)
		print("", file=txt)
		print("Model Summary (tree-depth2)", "\n", "------------------------------", sep="", file=txt)
		print("Overall accuracy = .", str(round(accuracy2, 3))[2:], "\n", sep="", file=txt)
		print("", file=txt)

		print("   (2)    Tree with depth 4", sep="", file=txt)
		print("ID, Actual class, tree-depth4 pred", "\n", "-----------------------------", sep="", file=txt)
		for k in range(len(data1_test)):
			print(k+1, ", ", int(data2_test[k, col_num]), ", ", np.array(list_output_class22, dtype='int')[k], sep="", file=txt)
		print("", file=txt)
		print("Confusion Matrix (tree-depth4)", "\n", "----------------------------------", file=txt)
		print(df_confusion_matrix_testing2, file=txt)
		print("", file=txt)
		print("Model Summary (tree-depth4)", "\n", "------------------------------", sep="", file=txt)
		print("Overall accuracy = .", str(round(accuracy22, 3))[2:], "\n", sep="", file=txt)
		print("", file=txt)

def hw11():
		start_fn() #C:/Users/YISS/Desktop/data_mining/hw/hw11
		categorical_col()
		class_col()
		training_data()
		testing_data()
		read_table_training()
		read_table_test()
		classification_decision_tree()
		save_as_file_classification()

#######################################################

hw11() # 19, 6
