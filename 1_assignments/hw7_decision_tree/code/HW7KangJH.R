#HW_7, 2018321086, Jaehun Kang

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
	sink('HW7KangJH_R_regression_output.txt')
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
	data_name2 <<- readline("Enter the test data file name: (if the test data set doesn't exist, enter the training data file name)")
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

	if (sum(data1=="?")>0){
		data1[data1=="?"] <<- NA
		data1_test[data1_test=="?"] <<- NA
		data1<<-data1
		data1_test<<-data1_test
		list_classes_cnt <<- table(data1[col_num])
		c1 = names(list_classes_cnt)
		list_classes <<- as.integer(c1)
		#K : number of classes
		K <<- length(list_classes)
	} else {
		data1<<-data1
		data1_test<<-data1_test
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
	class_0 <<- min(data1_y)

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
			list_output_class1 <<- cbind(list_output_class1, class_0)
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
			list_output_class2 <<- cbind(list_output_class2, class_0)
		} else {
			print("error!!")
		}
	}
	data1_test$output_logistic_testing = matrix(list_output_class2, dim(data1_test)[1], 1)
	data1_test<<-data1_test
	accuracy_logistic_testing <<- sum(data1_test[col_num] == data1_test['output_logistic_testing'])/dim(data1_test)[1]
}


pdf_normal_dist = function(value, m, s) {
	temp_val <<- (1/(sqrt(2*pi)*s)*exp(-(value-m)**2/(2*(s**2))))
}

classification_NaiveBayes = function(){
	name <<- "Naive_bayes"	
	threshold <<- 0.5

	data1_x <<- data1[-col_num]
	data1_y <<- data1[col_num]
	data1_test_x <<- data1_test[-col_num]
	data1_test_y <<- data1_test[col_num]
	n2 <<- dim(data1_test)[1]

	class_1 <<- max(data1_y)
	class_0 <<- min(data1_y)

	data2_x <<- cbind(data1_x)
	data2_y <<- (data1_y == class_1)*1
	data2_test_x <<- cbind(data1_test_x)
	data2_test_y <<- (data1_test_y == class_1)*1

	data3 <<- cbind(data2_x, data2_y)
	num_of_cols <<- dim(data3)[2]

	col_continuous <<- c()
	col_discrete <<- c()
	for (i in 1:num_of_cols){
		if (length(unique(data3[ , i])) > 10){
			col_continuous <<- cbind(col_continuous, i)
		} else {
			col_discrete <<- cbind(col_discrete, i)
		}
	}
	data3_cla1 <<- data3[data3[col_num] == 1, ]
	data3_cla0 <<- data3[data3[col_num] == 0, ]

	#training
	list_output_class1 <<- c()
	predicted_result1 <<- c()
	for (i in 1:n){
		#data3_cla1
		start_value1 <<- 1
		start_value2 <<- 1
		for (j in 1:(num_of_cols-1)){
			if (is.na(data3[i, j]) == 0){
				if (sum(col_continuous == j) >= 1){
					pdf_normal_dist(data3[i, j] , mean(data3_cla1[ , j]), sd(data3_cla1[ , j]))
					start_value1 <<- start_value1*temp_val
				} else{
					temp_val2 <<- as.numeric(summary(data3_cla1[ ,j] == data3[i, j])["TRUE"])/dim(data3_cla1)[1]
					start_value1 <<- start_value1*temp_val2
				}
			} else {
			}
		}
		start_value1 <<- start_value1*(dim(data3_cla1)[1]/dim(data3)[1])		

		#data3_cla0
		for (j in 1:(num_of_cols-1)){
			if (is.na(data3[i, j]) == 0){
				if (sum(col_continuous == j) >= 1){
					pdf_normal_dist(data3[i, j] , mean(data3_cla0[ , j]), sd(data3_cla0[ , j]))
					start_value2 <<- start_value2*temp_val
				} else{
					temp_val2 <<- as.numeric(summary(data3_cla0[ ,j] == data3[i, j])["TRUE"])/dim(data3_cla0)[1]
					start_value2 <<- start_value2*temp_val2
				}
			} else {
			}
		}
		start_value2 <<- start_value2*(dim(data3_cla0)[1]/dim(data3)[1])	

		temp_output <<- c()
		prob1 <<- start_value1/(start_value1 + start_value2)
		prob2 <<- start_value2/(start_value1 + start_value2)
		temp_output <<- cbind(temp_output, prob1, prob2)
		predicted_result1 <<- cbind(predicted_result1, prob1)

		if (temp_output[1] > temp_output[2]){
			list_output_class1 <<- cbind(list_output_class1, class_1)
		} else {
			list_output_class1 <<- cbind(list_output_class1, class_0)
		}
	}
	data1$output_Naive_bayes_training <<- matrix(list_output_class1, dim(data1)[1], 1)
	accuracy_Naive_bayes_training <<- sum(list_output_class1 == data1_y)/n

	#testing
	list_output_class2 <<- c()
	predicted_result2 <<- c()
	for (i in 1:dim(data1_test)[1]){
		#data3_cla1
		start_value1 <<- 1
		start_value2 <<- 1
		for (j in 1:(num_of_cols-1)){
			if (is.na(data1_test[i, j]) == 0){
				if (sum(col_continuous == j) >= 1){
					pdf_normal_dist(data1_test[i, j] , mean(data3_cla1[ , j]), sd(data3_cla1[ , j]))
					start_value1 <<- start_value1*temp_val
				} else{
					temp_val2 <<- as.numeric(summary(data3_cla1[ ,j] == data1_test[i, j])["TRUE"])/dim(data3_cla1)[1]
					start_value1 <<- start_value1*temp_val2
				}
			} else {
			}
		}
		start_value1 <<- start_value1*(dim(data3_cla1)[1]/dim(data3)[1])		

		#data3_cla0
		for (j in 1:(num_of_cols-1)){
			if (is.na(data1_test[i, j]) == 0){
				if (sum(col_continuous == j) >= 1){
					pdf_normal_dist(data1_test[i, j] , mean(data3_cla0[ , j]), sd(data3_cla0[ , j]))
					start_value2 <<- start_value2*temp_val
				} else{
					temp_val2 <<- as.numeric(summary(data3_cla0[ ,j] == data1_test[i, j])["TRUE"])/dim(data3_cla0)[1]
					start_value2 <<- start_value2*temp_val2
				}
			} else {
			}
		}
		start_value2 <<- start_value2*(dim(data3_cla0)[1]/dim(data3)[1])	

		temp_output <<- c()
		prob1 <<- start_value1/(start_value1 + start_value2)
		prob2 <<- start_value2/(start_value1 + start_value2)
		temp_output <<- cbind(temp_output, prob1, prob2)
		predicted_result2 <<- cbind(predicted_result2, prob1)

		if (temp_output[1] > temp_output[2]){
			list_output_class2 <<- cbind(list_output_class2, class_1)
		} else {
			list_output_class2 <<- cbind(list_output_class2, class_0)
		}
	}
	data1_test$output_Naive_bayes_testing <<- matrix(list_output_class2, dim(data1_test)[1], 1)
	accuracy_Naive_bayes_testing <<- sum(list_output_class2 == data1_test_y)/n
}

impurity = function(input_data_class){
	input_data_class_unique <<- unique(input_data_class)
	temp7 <<- c()
	for (i in 1:length(input_data_class_unique)){
		temp7 <<- cbind(temp7, sum(input_data_class == input_data_class_unique[i])/length(input_data_class))
	}
	temp8 <<- temp7**2
	temp9 <<- sum(temp8)
	imp_output <<- 1-temp9
	return(imp_output)
}

classification_one_level_decision_tree = function(){
	name <<- "1-level_decision_tree"	

	data1_x <<- data1[-col_num]
	data1_y <<- data1[col_num]
	data1_test_x <<- data1_test[-col_num]
	data1_test_y <<- data1_test[col_num]
	n2 <<- dim(data1_test)[1]

	class_1 <<- max(data1_y)
	class_0 <<- min(data1_y)

	data2_x <<- cbind(data1_x)
	data2_y <<- (data1_y == class_1)*1
	data2_test_x <<- cbind(data1_test_x)
	data2_test_y <<- (data1_test_y == class_1)*1

	data3 <<- cbind(data2_x, data2_y)
	num_of_cols <<- dim(data3)[2]

	#boundary, goodness_of_split
	boundary_gof_1 <<- data.frame(V1=double(), goodness_of_split=double())
	for (j in 1:(p-1)){
		input_x <<- matrix(data1_x[ , j], n, 1)
		input_y <<- matrix(data1_y[ , ], n, 1)
		input_data1 <<- cbind(input_x, input_y)
		input_data2 <<- input_data1[order(input_data1[, 1]), ]
		input_data_x_val <<- input_data2[ , 1]
		input_data_x_cla <<- input_data2[ , 2]
		input_data_x_unique <<- unique(input_data_x_val)

		#impurity_total(j_th col)
		imp_t <<- impurity(input_data_x_cla)
		#impurity_by_boundary
		boundary_list <<- list()
		for (i in 1:length(input_data_x_unique)-1){	
			a <<- (input_data_x_unique[i]+input_data_x_unique[i+1])/2
			boundary_list <<- append(boundary_list, a)
		}

		imp_list2 <<- data.frame(V1=double(), goodness_of_split=double()) #impurity_list
		for (k in 1:length(boundary_list)){
			boundary <<- as.numeric(boundary_list[k])
			left_side <<- as.matrix(input_data2[input_data_x_val <= boundary, ])
			right_side <<- as.matrix(input_data2[input_data_x_val > boundary, ])
			#since the # of row is one, dim of mat is (2,1). so, we have to tansfer its dim to (1,2).
			if (sum(dim(right_side) == c(2,1)) == 2) {
				right_side <<- t(right_side)
			} else {
				right_side <<- right_side
			}
			if (sum(dim(left_side) == c(2,1)) == 2) {
				left_side <<- t(left_side)
			} else {
				left_side <<- left_side
			}
			goodness_of_split <<- imp_t - impurity(left_side[, 2])*(dim(left_side)[1]/(dim(left_side)[1]+dim(right_side)[1])) - impurity(right_side[, 2])*(dim(right_side)[1]/(dim(left_side)[1]+dim(right_side)[1]))
			imp_list <<- c()
			imp_list <<- cbind(imp_list, boundary_list[k]) 	
			imp_list <<- cbind(imp_list, goodness_of_split)
			imp_list <<- as.data.frame(imp_list)
			imp_list2 <<- rbind(imp_list2, imp_list)
		}
		output1 <<- imp_list2[which.max(imp_list2[ , 2]), ]
		boundary_gof_1 <<- rbind(boundary_gof_1, output1)
	}

	#final_output (specific boundary)
	j = as.numeric(which.max(boundary_gof_1[ , 2]))
	axis_name <<- paste("x", as.character(j), sep="")
	input_x <<- as.matrix(data1_x[ , j])
	input_data1 <<- cbind(input_x, input_y)
	input_data2 <<- input_data1[order(input_data1[, 1]), ]
	input_data_x_val <<- input_data2[ , 1]
	input_data_x_cla <<- input_data2[ , 2]
	input_data_x_unique <<- unique(input_data_x_val)

	boundary = as.numeric(boundary_gof_1[j, 1])
	boundary <<- boundary
	left_side <<- as.matrix(input_data2[input_data_x_val <= boundary, ])
	right_side <<- as.matrix(input_data2[input_data_x_val > boundary, ])
	#since the # of row is one, dim of mat is (2,1). so, we have to tansfer its dim to (1,2).
	if (sum(dim(right_side) == c(2,1)) == 2) {
		right_side <<- t(right_side)
	} else {
		right_side <<- right_side
	}
	if (sum(dim(left_side) == c(2,1)) == 2) {
		left_side <<- t(left_side)
	} else {
		left_side <<- left_side
	}
	goodness_of_split <<- imp_t - impurity(left_side[, 2])*(dim(left_side)[1]/(dim(left_side)[1]+dim(right_side)[1])) - impurity(right_side[, 2])*(dim(right_side)[1]/(dim(left_side)[1]+dim(right_side)[1]))

	list_output_class1 <<- c()
	for (i in 1:n){
		if (data1[i, j] <= boundary) {
			list_output_class1 <<- cbind(list_output_class1, class_1)
		} else {
			list_output_class1 <<- cbind(list_output_class1, class_0)
		}
	}

	data1$output_1_level_decision_tree_training <<- matrix(list_output_class1, dim(data1)[1], 1)
	accuracy_1_level_decision_tree_training <<- sum(list_output_class1 == data1[ , p])/n

	#testing
	n2 <<- dim(data1_test_x)[1]
	test_input_x <<- matrix(data1_test_x[ , j], n2, 1)
	test_input_y <<- matrix(data1_test_y[ , ], n2, 1)
	test_input_data <<- cbind(test_input_x, test_input_y)
	test_input_data2 <<- test_input_data[order(test_input_data[, 1]), ]
	test_input_data_x_val <<- test_input_data2[ , 1]
	test_input_data_x_cla <<- test_input_data2[ , 2]
	test_input_data_x_unique <<- unique(test_input_data_x_val)

	list_output_class2 <<- c()
	for (i in 1:n2){
		if (data1_test[i, j] <= boundary) {
			list_output_class2 <<- cbind(list_output_class2, class_1)
		} else {
			list_output_class2 <<- cbind(list_output_class2, class_0)
		}
	}

	data1_test$output_1_level_decision_tree_testing <<- matrix(list_output_class2, dim(data1_test)[1], 1)
	accuracy_1_level_decision_tree_testing <<- sum(list_output_class2 == data1_test[ , p])/n2
}

classification = function(){
	num_of_class <<- dim(unique(data1[col_num]))[1]
	if (num_of_class > 2){
		cat("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA')")
		what <<- scan(n=1, quiet=TRUE)
	} else {
		cat("Select the classifier(1 = 'LDA' or 2 = 'QDA' or 3 = 'RDA' or 4 = 'Logistic' or 5 = 'Naive_bayes' or 6 = '1-level_decision_tree')")
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
	} else if (what == 5){
		what_to_do <<- 'output_Naive_bayes'
		classification_NaiveBayes()
	} else if (what == 6){
		what_to_do <<- 'output_1_level_decision_tree'
		classification_one_level_decision_tree()
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
	} else if (what == 5){
		sensitivity_Naive_bayes_training <<- df_confusion_matrix_training[2, 2]/(df_confusion_matrix_training[2, 1]+df_confusion_matrix_training[2, 2])
		specificity_Naive_bayes_training <<- df_confusion_matrix_training[1, 1]/(df_confusion_matrix_training[1, 1]+df_confusion_matrix_training[1, 2])
		sensitivity_Naive_bayes_testing <<- df_confusion_matrix_testing[2, 2]/(df_confusion_matrix_testing[2, 1]+df_confusion_matrix_testing[2, 2])
		specificity_Naive_bayes_testing <<- df_confusion_matrix_testing[1, 1]/(df_confusion_matrix_testing[1, 1]+df_confusion_matrix_testing[1, 2])	
	} else if (what == 6){
		sensitivity_1_level_decision_tree_training <<- df_confusion_matrix_training[2, 2]/(df_confusion_matrix_training[2, 1]+df_confusion_matrix_training[2, 2])
		specificity_1_level_decision_tree_training <<- df_confusion_matrix_training[1, 1]/(df_confusion_matrix_training[1, 1]+df_confusion_matrix_training[1, 2])
		sensitivity_1_level_decision_tree_testing <<- df_confusion_matrix_testing[2, 2]/(df_confusion_matrix_testing[2, 1]+df_confusion_matrix_testing[2, 2])
		specificity_1_level_decision_tree_testing <<- df_confusion_matrix_testing[1, 1]/(df_confusion_matrix_testing[1, 1]+df_confusion_matrix_testing[1, 2])	
	}
}

print_fn_classification = function(){
	output_file <<- paste("HW7KangJH_R_", name, "_output.txt", sep="")
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
	} else if (what==6) {
		cat("Tree Structure", "\n", sep="")
		cat("      Node 1: ", axis_name, " <= ", as.numeric(boundary), " (", sum(left_side[, 2] == max(data1_y))+sum(right_side[, 2] == max(data1_y)), ", ", sum(left_side[, 2] == min(data1_y))+sum(right_side[, 2] == min(data1_y)), ")\n", sep="")
		cat("        Node 2: ", class_1, " (", sum(left_side[ , 2] == max(data1_y)), ", ", sum(left_side[ , 2] == min(data1_y)), ")\n", sep="")
		cat("        Node 3: ", class_0, " (", sum(right_side[ , 2] == max(data1_y)), ", ", sum(right_side[ , 2] == min(data1_y)), ")\n\n", sep="")
	}
	if (what == 4 | what == 5){
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
	} else if (what==5){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_Naive_bayes_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_Naive_bayes_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_Naive_bayes_training, 3)), start=3, stop=5), sep="")
	} else if (what==6){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_1_level_decision_tree_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_1_level_decision_tree_training, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_1_level_decision_tree_training, 3)), start=3, stop=5), sep="")			
	} else {
		cat("error!!!")
	}
	cat("\n\n")

	if (what == 4 | what == 5){
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
	} else if (what==5){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_Naive_bayes_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_Naive_bayes_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_Naive_bayes_testing, 3)), start=3, stop=5), sep="")	
	} else if (what==6){
		cat("Overall accuracy = .", substr(as.character(round(accuracy_1_level_decision_tree_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Sensitivity = .", substr(as.character(round(sensitivity_1_level_decision_tree_testing, 3)), start=3, stop=5), "\n", sep="")
		cat("Specificity = .", substr(as.character(round(specificity_1_level_decision_tree_testing, 3)), start=3, stop=5), sep="")		
	} else {
		cat("error!!!")
	}
	sink()
}

################################
#2. Run
################################

reg_or_cla()
C:\Users\YISS\Desktop\data_mining\hw\hw7
2
pid.dat
2
pidtest.dat
2
8
6

