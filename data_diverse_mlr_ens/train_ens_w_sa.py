#! /usr/bin/python

# this script trains mlr with different weights to each sample

import sys
import math
from os import system
import numpy
import pickle

def grad_asc_w(w_old,X,t,eta,one_in_k_enc,sample_wts): # gradient ascent for the w's in MLR
        w_old = numpy.matrix(w_old)
	X = numpy.matrix(X)
	mat_exp_w_tran_x = numpy.exp(X*w_old)
        normalizing_factor = numpy.sum(mat_exp_w_tran_x,axis=1)
        norm_mat = numpy.tile(normalizing_factor,w_old.shape[1])
        probability = mat_exp_w_tran_x/norm_mat	

        grad_scales = numpy.multiply((probability-one_in_k_enc),numpy.tile(sample_wts,(1,probability.shape[1])))
	
        w_grads = grad_scales.transpose()*X
        w_grads = w_grads.transpose()
        rms_grads = numpy.sqrt(numpy.sum(numpy.square(w_grads)))
        w_grads = (1/rms_grads)*w_grads
        w_new = w_old - eta*w_grads
        return w_new

def get_accuracy(w,X,targets,sample_wts):
	w = numpy.matrix(w)
	X = numpy.matrix(X)
	mat_exp_w_tran_x = numpy.exp(X*w)
        normalizing_factor = numpy.sum(mat_exp_w_tran_x,axis=1)
        norm_mat = numpy.tile(normalizing_factor,w.shape[1])
        probability = mat_exp_w_tran_x/norm_mat # probability of all the classes

	obt_classes = numpy.argmax(probability,axis = 1 )
	obt_classes = obt_classes + numpy.ones((len(targets),1)) # the classes obtained

	equals = (obt_classes==targets)*1
	accuracy = (float)(numpy.sum(equals))/len(equals) # the accuracy

	cum_sum_log_prob_data = 0
        for i in range (0,len(numpy.unique(numpy.array(targets)))):
                wtd_class_probs = numpy.multiply(numpy.log(probability[:,i]),sample_wts)
                log_class_prob_t = wtd_class_probs[(targets==(i+1))]
                cum_sum_log_prob_data += numpy.sum(log_class_prob_t)

		
	return probability,obt_classes,accuracy,cum_sum_log_prob_data


def get_wts(probability,targets):
	prob_correct = numpy.zeros((X.shape[0],1))
	for i in range(0,probability.shape[0]):
		prob_correct[i] = probability[i,int(targets[i]-1)]
	print >> f1, 'prob correct: \n'
	print >> f1, prob_correct
	wts = numpy.ones(prob_correct.shape)-prob_correct
	#wts = prob_correct
	scaled_wts = (wts/numpy.sum(wts))*len(prob_correct)
	return scaled_wts

def get_acc_given_probs(probability,targets):
	obt_classes = numpy.argmax(probability,axis = 1 )
        obt_classes = obt_classes + numpy.ones((len(targets),1)) # the classes obtained
        equals = (obt_classes==targets)*1
	accuracy = (float)(numpy.sum(equals))/len(equals) # the accuracy
	return accuracy

def sim_anneal(w,X,targets,eta,one_in_k_enc,cur_feat_indic,sim_anl_prob,sample_wts):
	# first calculate accuracy for the current feature indicator
        print >> f1, "\n------- sim anneal iter begin ------------"
        cur_wts1 = w[(cur_feat_indic==1),:]
        cur_data1 = X[:,(cur_feat_indic==1)]
	
	# do one step of gradient accent
        cur_wts_new1 = grad_asc_w(cur_wts1,cur_data1,targets,eta,one_in_k_enc,sample_wts)
	probability,obt_classes,accuracy1,wtd_ll1 = get_accuracy(cur_wts_new1,cur_data1,targets,sample_wts)
	flip_probs =  numpy.random.random_sample(cur_feat_indic.shape)
	flip_probs_indic = (flip_probs > sim_anl_prob)
        cur_feat_indic_bool = (cur_feat_indic > 0)	
	
	for indic in range(0,len(cur_feat_indic)):
                if (flip_probs_indic[indic]):
                        cur_feat_indic_bool[indic] = not(cur_feat_indic_bool[indic])

	new_feat_indic = cur_feat_indic_bool * 1
        new_feat_indic[-1] = 1 # ensuring the bias term is always used


	# now grad asc using the new feat indic
	cur_wts2 = w[(new_feat_indic==1),:]
	cur_data2 = X[:,(new_feat_indic==1)]

	cur_wts_new2 = grad_asc_w(cur_wts2,cur_data2,targets,eta,one_in_k_enc,sample_wts)
	probability,obt_classes,accuracy2,wtd_ll2 = get_accuracy(cur_wts_new2,cur_data2,targets,sample_wts)

	print >> f1, "wtd ll 1:", wtd_ll1, "wtd ll 2:", wtd_ll2, "accuracy1:", accuracy1, "accuracy2:", accuracy2

#	if (wtd_ll1>=wtd_ll2):
#		print >> f1, "---sim anneal iter end---\n"
#		return cur_feat_indic,cur_wts_new1,wtd_ll1
#	else:
#		print >> f1,"modified feat_indic with ", numpy.sum(new_feat_indic), "features"
#                print >> f1,"------- sim anneal iter end ------------\n"
#		return new_feat_indic,cur_wts_new2,wtd_ll2

	if (accuracy1>=accuracy2):
               print >> f1, "---sim anneal iter end---\n"
               return cur_feat_indic,cur_wts_new1,wtd_ll1
       	else:
               print >> f1,"modified feat_indic with ", numpy.sum(new_feat_indic), "features"
               print >> f1,"------- sim anneal iter end ------------\n"
               return new_feat_indic,cur_wts_new2,wtd_ll2
	

def update_wt_matrix(wt_on_subset,complete_wt_mat,indic_vect):
        updated_wts = numpy.zeros(complete_wt_mat.shape)
        sub_clm_counter = 0
        for clm_counter in range(0,len(indic_vect)):
                if (indic_vect[clm_counter] == 0):
                        updated_wts[clm_counter,:] = complete_wt_mat[clm_counter,:]
                else:
                        updated_wts[clm_counter,:] = wt_on_subset[sub_clm_counter,:]
                        sub_clm_counter += 1
        return updated_wts

	
# main function with split
dir_name = sys.argv[1]
file_name = dir_name + 'data'
X = numpy.array(numpy.loadtxt(file_name,delimiter=','))

tuning_parameters_file = dir_name + 'tune_parameters'
tune_params = numpy.loadtxt(tuning_parameters_file,delimiter=',')
eta =  float(tune_params[0]) # eta for the grad due to mlr
maxiter = int(tune_params[1]) # number of maximum iterations 
sim_anl_prob =  float(sys.argv[2]) # probability to flip up feature selection bit
num_classifiers = int(tune_params[3]) # the number of sub classifiers to be trained
break_thr = float(tune_params[4])


if sim_anl_prob > 1:
	# split into 90% train and 10% test
	cv_indices = numpy.random.random_integers(1,10,size=(X.shape[0],1))
	f_cv = open(dir_name+'cv_cur','w+')
	pickle.dump(cv_indices,f_cv)
	f_cv.close()
else:
	# split into 90% train and 10% test
	f_cv = open(dir_name+'/cv_cur','r+')
	cv_indices = pickle.load(f_cv)
	f_cv.close()


train_data = X[numpy.nonzero(cv_indices>2)[0],:]
test_data = X[numpy.nonzero(cv_indices == 1)[0],:]
dev_data = X[numpy.nonzero(cv_indices == 2)[0],:]

train_mean = numpy.mean(train_data,axis=0)
train_std = numpy.std(train_data,axis=0)

X_train = (train_data-numpy.tile(train_mean,(train_data.shape[0],1)))/(numpy.tile(train_std,(train_data.shape[0],1))) # z normalizing X_train and test
X_test = (test_data-numpy.tile(train_mean,(test_data.shape[0],1)))/(numpy.tile(train_std,(test_data.shape[0],1)))
X_dev = (dev_data-numpy.tile(train_mean,(dev_data.shape[0],1)))/(numpy.tile(train_std,(dev_data.shape[0],1)))

X_train = numpy.hstack((X_train,numpy.ones((X_train.shape[0],1))))
X_test = numpy.hstack((X_test,numpy.ones((X_test.shape[0],1))))
X_dev = numpy.hstack((X_dev,numpy.ones((X_dev.shape[0],1)))) 

target_file = dir_name + 'targets'
targets = numpy.matrix(numpy.loadtxt(target_file,delimiter=','))
targets = targets.transpose()

train_targets = targets[numpy.nonzero(cv_indices > 2)[0],:]
test_targets = targets[numpy.nonzero(cv_indices == 1)[0],:]
dev_targets = targets[numpy.nonzero(cv_indices == 2)[0],:]

num_classes = len(numpy.unique(numpy.array(targets)))


# get one in k encoding
X = X_train
one_in_k_enc = numpy.zeros((X.shape[0],num_classes))
for i in range(0,len(train_targets)):
        one_in_k_enc[i,int(train_targets[i])-1]=1



#########################################################################################################
########################################################################################################
# First get results with SA
#########################################################################################################
########################################################################################################


###############################
#### parameter initialization
###############################


# making a python list where each element has initial wt for each classifier
clasf_wts = []
for clasf_iter in range(0,num_classifiers):
        wt_shape0 = X.shape[1]
        wt_shape1 = len(numpy.unique(numpy.array(train_targets)))
        wts = numpy.random.rand(wt_shape0,wt_shape1)
        clasf_wts.append(wts)    # randomly initialize each classifier with random wts

if sim_anl_prob > 1:
	log_file_name = dir_name + '/logfile_baseline'
	f1 = open(log_file_name,'w+')
else:
	log_file_name = dir_name + '/logfile_sa'
	f1 = open(log_file_name,'w+')


if sim_anl_prob > 1:
	f_w=open(dir_name+'weight_file', 'w+')
	pickle.dump(clasf_wts,f_w)
	f_w.close()
else:
	f2=open(dir_name+'new_wts', 'r+')
	clasf_wts = pickle.load(f2)
	f2.close()

sample_wts = numpy.ones((X.shape[0],1))
feature_dim = X.shape[1]
features_selected = numpy.tile(numpy.ones(feature_dim),(num_classifiers,1))


################################
#### optimization 
################################
for clasf_iter in range(0,num_classifiers):


	print >> f1,"----------------\n------------------\ntraining classifier ----------------\n------------------\n: ",clasf_iter
	print "training classifier : ",clasf_iter
	
        w = clasf_wts[clasf_iter]

	data_ll_iters = -1*float('Inf')*numpy.ones((num_classifiers,maxiter))
	data_acc_iters = numpy.zeros((num_classifiers,maxiter))
	store_wts = []
	store_feat_indic = []
	cur_feat_indic = features_selected[clasf_iter,:] 
		
	for conv_iters in range(0,maxiter):

		if (conv_iters <= -1 ):
			print >> f1,"------------------\nIteration number for convergence: ",conv_iters	
			# grad asc on w based on wts
			w_new = grad_asc_w(w,X,train_targets,eta,one_in_k_enc,sample_wts)		 		
			probability,obt_classes,accuracy,wtd_data_ll = get_accuracy(w_new,X,train_targets,sample_wts)
			data_ll_iters[clasf_iter,conv_iters] = wtd_data_ll
			data_acc_iters[clasf_iter,conv_iters] = accuracy
			w = w_new		
	
			print >> f1,"iter: ",clasf_iter, "accuracy: ",accuracy," data_ll: ",wtd_data_ll

			store_wts.append(w_new)
			store_feat_indic.append(cur_feat_indic)			


		else:  # start SA only after a few many number of iterations

			print >> f1,"------------------\nIteration number for convergence with SA: ",conv_iters
			cur_feat_indic[-1] = 1 # ensuring that the bias term is always used

			new_cur_feat_indic_sa, new_wt_cur_clasf_sa,wtd_data_ll = sim_anneal(w,X,train_targets,eta,one_in_k_enc,cur_feat_indic,sim_anl_prob,sample_wts)
			w = update_wt_matrix(new_wt_cur_clasf_sa,w,new_cur_feat_indic_sa)
			cur_feat_indic = new_cur_feat_indic_sa		

			data_cur_classifier = X[:,(cur_feat_indic == 1)]
			new_wt_cur_clasf_sa = w[(cur_feat_indic == 1),:]	
			probability,obt_classes,accuracy,wtd_data_ll = get_accuracy(new_wt_cur_clasf_sa,data_cur_classifier,train_targets,sample_wts)
			data_ll_iters[clasf_iter,conv_iters] = wtd_data_ll
			data_acc_iters[clasf_iter,conv_iters] = accuracy

			print >> f1,"iter: ",clasf_iter, "accuracy: ",accuracy," data_ll: ",wtd_data_ll

			store_wts.append(w)	
			store_feat_indic.append(cur_feat_indic)	
			
#			# convergence criterion	based on log likelihood	
#			current_lls_10 = data_ll_iters[clasf_iter,conv_iters-20:conv_iters-10]
#			current_lls = data_ll_iters[clasf_iter,conv_iters-9:conv_iters]
#			
#			max_cur_lls = numpy.max(current_lls)
#			max_cur_lls_10 = numpy.max(current_lls_10)
#
#			max_ll_pos = numpy.argmax(data_ll_iters[clasf_iter,:])
#			if (((max_cur_lls - max_cur_lls_10)/X.shape[0]) < break_thr):   # testing convergence based on wtd data ll
#				final_wts = store_wts[max_ll_pos]
#				final_feat_indics = store_feat_indic[max_ll_pos]	
#				print >> f1, "breaking at iteration", conv_iters+10
#				break
#
#
#		# if not broken take the best wts 
#		current_lls = data_ll_iters[clasf_iter,:]
#		max_ll_pos = numpy.argmax(current_lls)
#		final_wts = store_wts[max_ll_pos]
#		final_feat_indics = store_feat_indic[max_ll_pos]

			# convergence criterion based on accuracy
			max_acc_pos = numpy.argmax(data_acc_iters[clasf_iter,:])
			if (max_acc_pos < (conv_iters-10)):
                                final_wts = store_wts[max_acc_pos]
				final_feat_indics = store_feat_indic[max_acc_pos]

				data_cur_classifier = X[:,(final_feat_indics == 1)]
				final_wts_ds = final_wts[(final_feat_indics == 1),:]
				probability,obt_classes,accuracy,wtd_data_ll = get_accuracy(final_wts_ds,data_cur_classifier,train_targets,sample_wts)

                                print >> f1, "breaking at iteration", conv_iters, "accuracy: ", accuracy 
                                break

			# if not broken take the best wts
			max_acc_pos = numpy.argmax(data_acc_iters[clasf_iter,:])
			final_wts = store_wts[max_acc_pos]
			final_feat_indics = store_feat_indic[max_acc_pos]

	clasf_wts[clasf_iter] = final_wts # updating wts
	features_selected[clasf_iter,:] = final_feat_indics

	data_cur_classifier = X[:,(final_feat_indics == 1)]
        final_wts_ds = final_wts[(final_feat_indics == 1),:]
        probability,obt_classes,accuracy,wtd_data_ll = get_accuracy(final_wts_ds,data_cur_classifier,train_targets,sample_wts)
	print >> f1, "breaking at iteration check", conv_iters, "accuracy: ", accuracy

	# get cumulative probability over classifier trained till now
        for cur_clasf_trained in range(0,clasf_iter+1):
		
                cur_wts = clasf_wts[cur_clasf_trained]
		cur_feat_indic = features_selected[cur_clasf_trained,:]
		w_cur_clasf = cur_wts[(cur_feat_indic==1),:]
		data_cur_clasf = X[:,(cur_feat_indic==1)]
                probability_train,obt_classes,indiv_acc_train,wtd_data_ll = get_accuracy(w_cur_clasf,data_cur_clasf,train_targets,sample_wts)
                if cur_clasf_trained == 0:
                        cum_train_prob = probability_train
                else:
                        cum_train_prob += probability_train

	cum_train_prob = cum_train_prob/(clasf_iter+1)
        sample_wts = get_wts(cum_train_prob,train_targets) # get wts from prev classifier
	print >> f1,'sample_wts: \n ', sample_wts.transpose()

	
# give combined prediction on train set and test set

cum_train_prob = numpy.zeros((X.shape[0],num_classes))
cum_test_prob = numpy.zeros((X_test.shape[0],num_classes))
cum_dev_prob = numpy.zeros((X_dev.shape[0],num_classes))

print >> f1, 'indiv acc train indiv_acc test indiv_acc_dev cum_acc_train cum_acc_test cum_acc_dev'

all_dev_acc = numpy.zeros((1,num_classifiers))
all_test_acc = numpy.zeros((1,num_classifiers))
indiv_clf_dev_acc = numpy.zeros((1,num_classifiers))
indiv_clf_test_acc = numpy.zeros((1,num_classifiers))

for clasf_iter in range(0,num_classifiers):
	wts = clasf_wts[clasf_iter]
	feat_indic = features_selected[clasf_iter,:]
	wts_ds = wts[(feat_indic==1),:]

	X_train_ds = X[:,(feat_indic==1)]
	probability_train,obt_classes,indiv_acc_train,wtd_data_ll = get_accuracy(wts_ds,X_train_ds,train_targets,numpy.ones((X.shape[0],1)))

	X_test_ds = X_test[:,(feat_indic==1)]
	probability_test,obt_classes,indiv_acc_test,wtd_data_ll = get_accuracy(wts_ds,X_test_ds,test_targets,numpy.ones((X_test.shape[0],1)))

	X_dev_ds = X_dev[:,(feat_indic==1)]
	probability_dev,obt_classes,indiv_acc_dev,wtd_data_ll = get_accuracy(wts_ds,X_dev_ds,dev_targets,numpy.ones((X_dev.shape[0],1)))

	if clasf_iter == 0:
		cum_train_prob = numpy.log(probability_train)
		cum_test_prob = numpy.log(probability_test)
		cum_dev_prob = numpy.log(probability_dev)
	else:
		cum_train_prob += numpy.log(probability_train)
		cum_test_prob += numpy.log(probability_test)
		cum_dev_prob += numpy.log(probability_dev)

	cum_acc_train = get_acc_given_probs(cum_train_prob,train_targets)	
	cum_acc_test = get_acc_given_probs(cum_test_prob,test_targets)
	cum_acc_dev = get_acc_given_probs(cum_dev_prob,dev_targets)

	print >> f1, indiv_acc_train, indiv_acc_test, indiv_acc_dev, cum_acc_train, cum_acc_test, cum_acc_dev
	all_dev_acc[0,clasf_iter] = cum_acc_dev
	all_test_acc[0,clasf_iter] = cum_acc_test
	indiv_clf_dev_acc[0,clasf_iter] = indiv_acc_dev
	indiv_clf_test_acc[0,clasf_iter] = indiv_acc_test

f_wt_indics = dir_name + 'selected_feat_mat'
numpy.savetxt(f_wt_indics,features_selected,delimiter=',')

# get best test accuracy based on best dev accuracy
best_cum_pos = numpy.argmax(all_dev_acc)
best_indiv_pos = numpy.argmax(indiv_clf_dev_acc)

best_cum_acc = all_test_acc[0,best_cum_pos]
best_indiv_acc = indiv_clf_test_acc[0,best_indiv_pos]

first_clsf_acc = indiv_clf_test_acc[0,0]

print >> f1, '\n\n'
print >> f1, 'best_cum_tuned :', best_cum_acc,' best_indiv_tuned :', best_indiv_acc,'first clf acc tuned: ', first_clsf_acc


# get best test accuracy based on oracle parameter
best_cum_pos = numpy.argmax(all_test_acc)
best_indiv_pos = numpy.argmax(indiv_clf_test_acc)

best_cum_acc_o = all_test_acc[0,best_cum_pos]
best_indiv_acc_o = indiv_clf_test_acc[0,best_indiv_pos]

first_clsf_acc = indiv_clf_test_acc[0,0]

print >> f1, '\n\n'
print >> f1, 'best_cum_oracle :', best_cum_acc,' best_indiv_oracle :', best_indiv_acc,'first clf acc oracle: ', first_clsf_acc



if sim_anl_prob > 1:
	f_new_wts = open(dir_name+'new_wts','w+')
	pickle.dump(clasf_wts,f_new_wts)
	f_new_wts.close()

	result_file_name = dir_name + '/result_file'
	f3 = open(result_file_name,'a+')
	print >> f3, best_cum_acc_o, best_indiv_acc_o, best_cum_acc, best_indiv_acc, first_clsf_acc

else:
	result_file_name = dir_name + '/result_file_sa'
        f3 = open(result_file_name,'a+')
        print >> f3, best_cum_acc_o,best_indiv_acc_o,best_cum_acc, best_indiv_acc, first_clsf_acc

