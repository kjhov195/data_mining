#HW_5, 2018321086, Jaehun Kang

##############################################################################
#####R_code####################################################################
##############################################################################

install.packages("rgl")
library(rgl)

################################
#1. defining functions
################################
start_fn = function(){
	rm(list=ls())	#clear the old variables
	cat("what is the path to the data?") 
	path <<- scan(n=1, what="character", quiet=TRUE)
	setwd(path)
}

which_analysis = function(){
	cat("which analysis do you want?(1 = regression or 2 = classification)")
	analysis <<- scan(n=1, quiet=TRUE)
}

reg_or_cla = function(){
	start_fn()
	which_analysis()
	if (analysis == 1) {
		input_data()
		read_table()
		response_var_col()
		multiple_linear_regression()
		print_fn_reg()		
	} else if (analysis == 2){
		training_data()
		read_table_training()
		testing_data()
		read_table_test()
		class_col()
		classification_data_preprocessing()
		classification()
		get_confusion_matrix()
		print_fn_classification()
	} else {
		print("error!!!")
	}
}

################################
#1-1. multiple_linear_regression
################################
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

#which column the response variable is recorded
response_var_col = function(){
	cat("Select the column num of response variable(ex. 1, 2, 3, ...)")
	col_num  <<- scan(n=1, quiet=TRUE)
}

#fitting model
multiple_linear_regression = function(){
	#data: dataframe with no constant column
	n <<- dim(data1)[1]
	p <<- dim(data1)[2]
	#data2: dataframe with constant column
	data2 <<- cbind(constant=1, data1)
	df_Y <<- data2[col_num+1]
	df_X <<- data2[-(col_num+1)]
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
print_fn_reg = function(){
	sink('HW5KangJH_R_regression_output.txt')
	cat("Coefficients", "\n", "-------------", "\n",sep="")
	cat("Constant: ", b_vector[1], "\n", sep="")
	for (i in 2:len_b){
		cat("Beta", i-1, ":  ", b_vector[i], "\n", sep="")
	}
	cat("\n")
  	cat("ID, Actual values, Fitted values", "\n", "--------------------------------", "\n")
	for (i in 1:n){
		SLR_model_prediction(i)
		cat(i, ", ", data2[i, col_num+1], ", ", y_hat, "\n", sep="") 	
	}
	cat("\n")
  	cat("Model Summary", "\n", "-------------", "\n",sep="")
	cat("R-square = ", R_square, "\n")
	cat("MSE = ", MSE)
	sink()
}	

################################
#1-2. classification(LDA, QDA, RDA, Logistic)
################################

training_data = function(){
	data_name <<- readline("Enter the training data file name: ")
}

testing_data = function(){
	data_name2 <<- readline("Enter the test data file name: ")
}

read_table_training = function(){
	cat("Select the training data coding format(1 = 'a b c' or 2 = 'a,b,c': )") 
	fm = scan(n=1, quiet=TRUE)
	if(fm==1) {
		form=""
	} else {
		form=","
	}
	data1 <<- read.table(data_name, sep=form) 
}

read_table_test = function(){
	cat("Select the test data coding format(1 = 'a b c' or 2 = 'a,b,c': )") 
	fm = scan(n=1, quiet=TRUE)
	if(fm==1) {
		form=""
	} else {
		form=","
	}
	data1_test <<- read.table(data_name2, sep=form) 
}

#the number of the class column
class_col = function(){
	cat("Select the number of the class column(ex. 1, 2, 3, ...)")
	col_num <<- scan(n=1, quiet=TRUE)
	#col_num <<- col_num-1 #for python...
}

#data preprocessing & getting Sp
classification_data_preprocessing = function(){
	n <<- dim(data1)[1]
	p <<- dim(data1)[2]
	list_classes_cnt <<- table(data1[col_num])
	c1 = names(list_classes_cnt)
	list_classes <<- as.integer(c1)
	#K : number of classes
	K <<- length(list_classes)
	#p(w_k)
	list_p <<- list_classes_cnt/sum(list_classes_cnt)
	#Sp
	temp6 = matrix(rep(0, len=(p-1)**2), nrow = p-1)
	list_S <<- c()
	for (k in 1:K){
		n_k = n_k = list_classes_cnt[k]
		data1_k = data1[data1[col_num] == k, ]
		data1_k_x = data1_k[-col_num]
		list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			list_tempk = list()
		temp5_sum = matrix(rep(0, len=(p-1)**2), nrow = p-1)
		for (i in 1:n_k){
			list_xk_temp = matrix(data1_k_x[i, ], p-1, 1)
			list_xk_temp = matrix(as.double(list_xk_temp))
			mat7 = list_xk_temp - list_xk_bar
			temp5 = mat7%*%t(mat7)
			temp5_sum = temp5_sum + temp5
		}
		S_k <<- temp5_sum/(n_k-1)
		list_S <<- append(list_S, list(S_k))
		temp6 = temp6 + (list_classes_cnt[k]-1)*S_k
	}
	Sp <<- 1/sum(list_classes_cnt)*temp6
	Sp_inv <<- solve(Sp)
}

classification_LDA = function(){
	name<<-"LDA"
	data1_x <<- data1[-col_num]
	data1_test_x <<- data1_test[-col_num]
	#training
	list_output_class <<- c()
	for (i in 1:dim(data1_x)[1]){
		x_vec = matrix(data1_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			d_k_x = t(list_xk_bar)%*%Sp_inv%*%x_vec - 1/2*t(list_xk_bar)%*%Sp_inv%*%list_xk_bar + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1$output_LDA_training <<- matrix(list_output_class, dim(data1)[1], 1)
	data1<<-data1
	accuracy_LDA_training <<- sum(data1[col_num] == data1['output_LDA_training'])/dim(data1)[1]
	#testing
	list_output_class <<- c()
	for (i in 1:dim(data1_test_x)[1]){
		x_vec = matrix(data1_test_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			data1_k_x$output_LDA_training = NULL
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			d_k_x = t(list_xk_bar)%*%Sp_inv%*%x_vec - 1/2*t(list_xk_bar)%*%Sp_inv%*%list_xk_bar + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1_test$output_LDA_testing = matrix(list_output_class, dim(data1_test)[1], 1)
	data1_test<<-data1_test
	accuracy_LDA_testing <<- sum(data1_test[col_num] == data1_test['output_LDA_testing'])/dim(data1_test)[1]
}

classification_QDA = function(){
	name<<-"QDA"
	data1_x <<- data1[-col_num]
	data1_test_x <<- data1_test[-col_num]
	#training
	list_output_class <<- c()
	for (i in 1:dim(data1_x)[1]){
		x_vec = matrix(data1_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			d_k_x = -1/2*log(det(matrix(unlist(list_S[k]), p-1, p-1))) -1/2*t(x_vec-list_xk_bar)%*%(solve(matrix(unlist(list_S[k]), p-1, p-1)))%*%(x_vec-list_xk_bar) + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1$output_QDA_training = matrix(list_output_class, dim(data1)[1], 1)
	data1<<-data1
	accuracy_QDA_training <<- sum(data1[col_num] == data1['output_QDA_training'])/dim(data1)[1]
	#testing
	list_output_class <<- c()
	for (i in 1:dim(data1_test_x)[1]){
		x_vec = matrix(data1_test_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			data1_k_x$output_QDA_training = NULL
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			d_k_x = -1/2*log(det(matrix(unlist(list_S[k]), p-1, p-1))) -1/2*t(x_vec-list_xk_bar)%*%(solve(matrix(unlist(list_S[k]), p-1, p-1)))%*%(x_vec-list_xk_bar) + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1_test$output_QDA_testing = matrix(list_output_class, dim(data1_test)[1], 1)
	data1_test<<-data1_test
	accuracy_QDA_testing <<- sum(data1_test[col_num] == data1_test['output_QDA_testing'])/dim(data1_test)[1]
}


classification_RDA = function(){
	name<<-"RDA"
	#finding alpha, gamma
	data1_x <<- data1[-col_num]
	data1_test_x <<- data1_test[-col_num]
	mat_sigma <<- diag(p-1)*mean(diag(Sp))
	list_RDA_1 <<- c()
	alpha_val <<- -0.05
	for (i in 0:20){
		alpha_val <<- alpha_val + 0.05
		gamma_val <<- -0.05
		for (j in 0:20){
			gamma_val <<- gamma_val + 0.05
			list_output_class2 <<- c()
			for (h in 1:dim(data1_test_x)[1]){
				x_vec = matrix(data1_test_x[h, ], p-1, 1)
				x_vec = as.double(x_vec)
				list_discriminant <<- c()
				for (k in 1:K){
					n_k = list_classes_cnt[k]
					data1_k = data1[data1[col_num] == k, ]
					data1_k_x = data1_k[-col_num]
					data1_k_x$output_RDA_testing = NULL
					list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
					Sk_alpha_gamma = alpha_val*matrix(unlist(list_S[k]), p-1, p-1)+(1-alpha_val)*(gamma_val*Sp + (1-gamma_val)*mat_sigma)
					d_k_x = -1/2*log(det(Sk_alpha_gamma)) -1/2*t(x_vec-list_xk_bar)%*%(solve(Sk_alpha_gamma))%*%(x_vec-list_xk_bar) + log(list_p[k])
					list_discriminant <<- cbind(list_discriminant, d_k_x)
				}
				output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
				list_output_class2 <<- cbind(list_output_class2, output_class)
			}
			data1_test$output_RDA_testing = matrix(list_output_class2, dim(data1_test)[1], 1)
			data1_test<<-data1_test
			accuracy_RDA_testing <<- sum(data1_test[col_num] == data1_test['output_RDA_testing'])/dim(data1_test)[1]
			list_RDA_1 <<- cbind(list_RDA_1, c(alpha_val, gamma_val, accuracy_RDA_testing))
			data1_test$output_RDA_testing = NULL
			cat("Calculating..: ", round(((21*i+j)/(21*21))*100, 2), "%\n", sep="")
		}
	}
	cat("Calculating..: 100%\n", sep="")
	alpha_val <<- list_RDA_1[1, as.double(which(list_RDA_1 == max(list_RDA_1[3, ]), arr.ind=T)[1, 2])]
	gamma_val <<- list_RDA_1[2, as.double(which(list_RDA_1 == max(list_RDA_1[3, ]), arr.ind=T)[1, 2])]

	#training
	data1_x <<- data1[-col_num]
	data1_test_x <<- data1_test[-col_num]
	list_output_class <<- c()
	for (i in 1:dim(data1_x)[1]){
		x_vec = matrix(data1_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			Sk_alpha_gamma = alpha_val*matrix(unlist(list_S[k]), p-1, p-1)+(1-alpha_val)*(gamma_val*Sp + (1-gamma_val)*mat_sigma)
			d_k_x = -1/2*log(det(Sk_alpha_gamma)) -1/2*t(x_vec-list_xk_bar)%*%(solve(Sk_alpha_gamma))%*%(x_vec-list_xk_bar) + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1$output_RDA_training = matrix(list_output_class, dim(data1)[1], 1)
	data1<<-data1
	accuracy_RDA_training <<- sum(data1[col_num] == data1['output_RDA_training'])/dim(data1)[1]

	#testing
	list_output_class <<- c()
	for (i in 1:dim(data1_test_x)[1]){
		x_vec = matrix(data1_test_x[i, ], p-1, 1)
		x_vec = as.double(x_vec)
		list_discriminant <<- c()
		for (k in 1:K){
			n_k = list_classes_cnt[k]
			data1_k = data1[data1[col_num] == k, ]
			data1_k_x = data1_k[-col_num]
			data1_k_x$output_RDA_training = NULL
			list_xk_bar = matrix(colSums(data1_k_x)/n_k, p-1, 1)
			Sk_alpha_gamma = alpha_val*matrix(unlist(list_S[k]), p-1, p-1)+(1-alpha_val)*(gamma_val*Sp + (1-gamma_val)*mat_sigma)
			d_k_x = -1/2*log(det(Sk_alpha_gamma)) -1/2*t(x_vec-list_xk_bar)%*%(solve(Sk_alpha_gamma))%*%(x_vec-list_xk_bar) + log(list_p[k])
			list_discriminant <<- cbind(list_discriminant, d_k_x)		
		}
		output_class <<- which(list_discriminant == max(list_discriminant), arr.ind = TRUE)[2]
		list_output_class <<- cbind(list_output_class, output_class)
	}
	data1_test$output_RDA_testing = matrix(list_output_class, dim(data1_test)[1], 1)
	data1_test<<-data1_test
	accuracy_RDA_testing <<- sum(data1_test[col_num] == data1_test['output_RDA_testing'])/dim(data1_test)[1]

	#result
	list_RDA_temp2 <<- c(alpha_val, gamma_val, accuracy_RDA_training, accuracy_RDA_testing)
}

sigmoid_fn = function(xb) {
	sigmoid_val <<- (1/(1+exp(-xb)))
}

minus_log_likelyhood = function(beta_vec){
	part_left <<- t(input_y)%*%input_x%*%beta_vec
	part_right <<- sum(log(1/(1+exp((input_x%*%beta_vec)))))
	#log_likelyhood = part_left+part_right
	#likelyhood = part_left+part_right
	likelyhood = exp(part_left+part_right)
	log_likelyhood <<- log(likelyhood)
	-log_likelyhood
}

mle_maximize = function(x, y){
	input_x <<- as.matrix(x)
	input_y <<- as.matrix(y)
	result_minimize <<- nlm(minus_log_likelyhood, matrix(0, p, 1))	
} 

classification_logistic = function(){
	name <<- "logistic"
	threshold <<- 0.5

	data1_x <<- data1[-col_num]
	data1_y <<- data1[col_num]
	data1_test_x <<- data1_test[-col_num]
	data1_test_y <<- data1_test[col_num]
	n2 <<- dim(data1_test)[2]

	class_1 <<- max(data1_y)
	class_2 <<- min(data1_y)

	data2_x <<- as.matrix(cbind(constant=1, data1_x))
	data2_y <<- (data1_y == class_1)*1
	data2_test_x <<- cbind(constant=1, data1_test_x)
	data2_test_y <<- (data1_test_y == class_1)*1

	#training
	mle_maximize(data2_x, data2_y)
	beta_vec <<- result_minimize$estimate
	xb <<- as.matrix(data2_x)%*%beta_vec
	sigmoid_fn(xb)
	predicted_result1 <<-sigmoid_val
	result_class <<- (predicted_result1>threshold)*1

	list_output_class1 <<- c()
	for (i in 1:dim(data1_x)[1]){
		if (result_class[i] == 1){
			list_output_class1 <<- cbind(list_output_class1, class_1)
		} else if (result_class[i] == 0){
			list_output_class1 <<- cbind(list_output_class1, class_2)
		} else {
			print("error!!")
		}
	}
	data1$output_logistic_training = matrix(list_output_class1, dim(data1)[1], 1)
	data1<<-data1
	accuracy_logistic_training <<- sum(data1[col_num] == data1['output_logistic_training'])/dim(data1)[1]

	#testing
	xb <<- as.matrix(data2_test_x)%*%beta_vec
	sigmoid_fn(xb)
	predicted_result2 <<-sigmoid_val
	result_class <<- (predicted_result2>threshold)*1

	list_output_class2 <<- c()
	for (i in 1:dim(data1_test_x)[1]){
		if (result_class[i] == 1){
			list_output_class2 <<- cbind(list_output_class2, class_1)
		} else if (result_class[i] == 0){
			list_output_class2 <<- cbind(list_output_class2, class_2)
		} else {
			print("error!!")
		}
	}
	data1_test$output_logistic_testing = matrix(list_output_class2, dim(data1_test)[1], 1)
	data1_test<<-data1_test
	accuracy_logistic_testing <<- sum(data1_test[col_num] == data1_test['output_logistic_testing'])/dim(data1_test)[1]
}

classification = function(){
	num_of_class <<- dim(unique(data1[col_num]))[1]
	if (num_of_class > 2){
		cat("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')")
		what <<- scan(n=1, quiet=TRUE)
	} else {
		cat("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA' or 4 = 'Logistic')")
		what <<- scan(n=1, quiet=TRUE)
	}

	if (what == 1){
		what_to_do <<- 'output_LDA'
		classification_LDA()
	} else if (what == 2){
		what_to_do <<- 'output_QDA'
		classification_QDA()
	} else if (what == 3){
		what_to_do <<- 'output_RDA'
		classification_RDA()
	} else if (what == 4){
		what_to_do <<- 'output_logistic'
		classification_logistic()
	} else {
		cat('error!!! Select the classifier again!!!')
	}
}

get_confusion_matrix = function(){
	confusion_matrix_training <<- c()
	for (i in 1:length(list_classes)){
		for (j in 1:length(list_classes)){
			str_1 <<- paste(what_to_do, "_training", sep="")
			val_1 = sum(data1[ , p] == i & data1[ , str_1] == j)
			confusion_matrix_training <<- cbind(confusion_matrix_training, val_1)
		}
	}
	df_confusion_matrix_training = data.frame(t(matrix(confusion_matrix_training, nrow=length(list_classes))))
	names(df_confusion_matrix_training) = seq(1, K, 1)
	df_confusion_matrix_training <<- df_confusion_matrix_training 

	confusion_matrix_testing <<- c()
	for (i in 1:length(list_classes)){
		for (j in 1:length(list_classes)){
			str_2 <<- paste(what_to_do, "_testing", sep="")
			val_2 = sum(data1_test[ , p] == i & data1_test[ , str_2] == j)
			confusion_matrix_testing<<- cbind(confusion_matrix_testing, val_2)
		}
	}
	df_confusion_matrix_testing = data.frame(t(matrix(confusion_matrix_testing, nrow=length(list_classes))))
	names(df_confusion_matrix_testing) = seq(1, K, 1)
	df_confusion_matrix_testing <<- df_confusion_matrix_testing

	if (what == 4){
		sensitivity_logistic_training <<- df_confusion_matrix_training[2, 2]/(df_confusion_matrix_training[2, 1]+df_confusion_matrix_training[2, 2])
		specificity_logistic_training <<- df_confusion_matrix_training[1, 1]/(df_confusion_matrix_training[1, 1]+df_confusion_matrix_training[1, 2])
		sensitivity_logistic_testing <<- df_confusion_matrix_testing[2, 2]/(df_confusion_matrix_testing[2, 1]+df_confusion_matrix_testing[2, 2])
		specificity_logistic_testing <<- df_confusion_matrix_testing[1, 1]/(df_confusion_matrix_testing[1, 1]+df_confusion_matrix_testing[1, 2])
	}
}

print_fn_classification = function(){
	output_file <<- paste("HW5KangJH_R_", name, "_output.txt", sep="")
	sink(file = output_file, append=FALSE)
	if (what == 3){
		cat("alpha : ", alpha_val, "\ngamma : ", gamma_val, "\n\n", sep="")
		X <<- list_RDA_1[1, ]
		Y <<- list_RDA_1[2, ]
		Z <<- list_RDA_1[3, ]
		plot3d(X, Y, Z, type='s', size=0.8, col = c("green"))
		bgplot3d({
			plot.new()
			title(main = "accuracy_RDA_testing", line = 2)
			mtext(side = 1, "X(alpha), Y(gamma), Z(test_accuracy)", line = 3.5)
		})
	}
	if (what == 4){
		cat("ID, Actual class, Resub pred, Pred Prob", "\n", "-----------------------------", "\n", sep="")
		for (k in 1:dim(data1)[1]){
			cat(k, ", ", data1[k, col_num], ", ", data1[k, str_1], ", ", round(predicted_result1[k], 3), "\n",  sep="")
	}
	} else {
		cat("ID, Actual class, Resub pred", "\n", "-----------------------------", "\n", sep="")
		for (k in 1:dim(data1)[1]){
			cat(k, ", ", data1[k, col_num], ", ", data1[k, str_1], "\n",  sep="")
		}
	}
		
	cat("\n")
	cat("Confusion Matrix (Resubstitution)", "\n", "----------------------------------", "\n")
	print(df_confusion_matrix_training)
	cat("\n")
	cat("Model Summary (Resubstitution)", "\n", "------------------------------", "\n", sep="")
	if (what==1){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_LDA_training, 3)), start=3, stop=5), sep="")
	} else if (what==2){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_QDA_training, 3)), start=3, stop=5), sep="")
	} else if (what==3){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_RDA_training, 3)), start=3, stop=5), sep="")
	} else if (what==4){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_logistic_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_logistic_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_logistic_training, 3)), start=3, stop=5), sep="")
	} else {
		cat("error!!!")
	}
	cat("\n\n")

	if (what == 4){
		cat("ID, Actual class, Test pred, Pred Prob", "\n", "-----------------------------", "\n", sep="")
		for (k in 1:dim(data1_test)[1]){
			cat(k, ", ", data1_test[k, col_num], ", ", data1_test[k, str_2], ", ", round(predicted_result2[k], 3), "\n",  sep="")
	}
	} else {
		cat("ID, Actual class, Test pred", "\n", "-----------------------------", "\n", sep="")
		for (k in 1:dim(data1_test)[1]){
			cat(k, ", ", data1_test[k, col_num], ", ", data1_test[k, str_2], "\n",  sep="")
		}
	}
	cat("\n")

	cat("Confusion Matrix (Test)", "\n", "----------------------------------", "\n")
	print(df_confusion_matrix_testing)
	cat("\n")
	cat("Model Summary (Test)", "\n", "------------------------------", "\n", sep="")
	if (what==1){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_LDA_testing, 3)), start=3, stop=5), sep="")
	} else if (what==2){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_QDA_testing, 3)), start=3, stop=5), sep="")
	} else if (what==3){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_RDA_testing, 3)), start=3, stop=5), sep="")
	} else if (what==4){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_logistic_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_logistic_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_logistic_testing, 3)), start=3, stop=5), sep="")
	} else {
		cat("error!!!")
	}
	sink()
}

################################
#2. Run
################################

reg_or_cla()
C:\Users/YISS/Desktop/data_mining/hw/hw5
2
pid.dat
2
pidtest.dat
2
8
4
