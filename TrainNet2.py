#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:23:32 2019

@author: labadmin
"""

from __future__ import division
from __future__ import print_function
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers.graph import GraphConvolution
import time
from utils import *
import utils2 as Ut2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from graph_fusing_layer import GraphFlusing
import utils_mine as Reader
def avg_std_log(train_accuracy, val_accuracy, test_accuracy):
    # average
    train_avg_acc = np.mean(train_accuracy)
    val_avg_acc = np.mean(val_accuracy)
    test_avg_acc = np.mean(test_accuracy)

    # std
    train_std_acc = np.std(train_accuracy)
    val_std_acc = np.std(val_accuracy)
    test_std_acc = np.std(test_accuracy)

    print('Average accuracies:')
    print('train_avg: ', train_avg_acc, '±', train_std_acc)
    print('val_avg: ', val_avg_acc, '±', val_std_acc)
    print('test_avg: ', test_avg_acc, '±', test_std_acc)
    print()
    print()
    return train_avg_acc, train_std_acc, val_avg_acc, val_std_acc, test_avg_acc, test_std_acc

def Compute_Matrix(A, MAX_DEGREE=3, SYM_NORM = True):
    L = Ut2.normalized_laplacian(A, SYM_NORM)
    L_scaled = Ut2.rescale_laplacian(L)
    T_k = Ut2.chebyshev_polynomial(L_scaled, MAX_DEGREE)
    return T_k

def MineModel(feature, support, ACT, weight_deacy, G, G1):
    X_in1 = Input(shape=(feature.shape[1],))
    H = Dropout(0.3)(X_in1)
    
    H0 = GraphConvolution(16, support, activation=ACT, kernel_regularizer=l2(weight_deacy))([H]+G)
    H0 = Dropout(0.3)(H0)
    Y0 = GraphConvolution(3, support, activation=ACT)([H0]+G)

    H1 = GraphConvolution(16, support, activation=ACT, kernel_regularizer=l2(weight_deacy))([H]+G1)
    H1 = Dropout(0.3)(H1)
    Y1 = GraphConvolution(3, support, activation=ACT)([H1]+G1)

    out = GraphFlusing('attention',activation='softmax')([Y0,Y1])#Activation('softmax')(add([Y0,Y1]))#
    
    # Compile model
    model = Model(inputs=[X_in1] +G +G1, outputs=out)
    model.compile(loss='categorical_crossentropy',weighted_metrics=['acc'], optimizer=Adam(lr=0.005))
    return model
        


def Training():
    # Loading data
    sparsity_threshold = 0.5
    age_adj, gender_adj, fdg_adj, apoe_adj, mixed_adj, features, all_labels, one_hot_labels, node_weights, dense_features = \
        load_tadpole_data(sparsity_threshold)
    adj_dict = {'age': age_adj, 'gender': gender_adj, 'fdg': fdg_adj, 'apoe': apoe_adj, 'mixed': mixed_adj}
    num_class = 3    
    num_folds = 10
    MAX_DEGREE = 3  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    weight_deacy = 5e-4
    ACT = 'elu'
    feature=dense_features#features[2]        
    # Normalize X
    feature = Reader.preprocess_features((sp.csr_matrix(feature, dtype=np.float32).todense()))
    print(feature.shape)
    
    adj1 = sp.csr_matrix(mixed_adj, dtype=np.float32)
    adj2 = np.load('W_G2_2.npy')
    adj2 = sp.csr_matrix(adj2/4., dtype=np.float32)
          
    print('Using Chebyshev polynomial basis filters...')
    T_k0 = Compute_Matrix(adj1) 
    T_k1 = Compute_Matrix(adj2) 
    graph0 = [feature]+T_k0+T_k1
    support = MAX_DEGREE + 1
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
    G1 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
    
    # index of fold for validation set and test set  
    val_part = 0
    test_part = 1    
    epochs = 1500
    Final_Score = []
    # Num_folds cross validation
    for fold in range(1):#num_folds
        print('Starting fold {}'.format(fold + 1))
          
        model = MineModel(feature, support, ACT, weight_deacy, G, G1)
        # Create model
        num_nodes = dense_features.shape[0]
        fold_size = int(num_nodes / num_folds)
        
        # shape of features
        print('whole features shape: ', dense_features.shape)
    
        # rotating folds of val and test
        val_part = (val_part + 1) % 10
        test_part = (test_part + 1) % 10
        print('fold_{},val_ind_{},test_ind_{} '.format(fold + 1, val_part, test_part))
    
        # defining train, val and test mask
        train_mask = np.ones((num_nodes,), dtype=np.bool)
        val_mask = np.zeros((num_nodes,), dtype=np.bool)
        test_mask = np.zeros((num_nodes,), dtype=np.bool)
        train_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 0
        train_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 0
        val_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 1
        test_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 1
    
        # defining train, val and test labels
        y_train = np.zeros(one_hot_labels.shape)
        y_val = np.zeros(one_hot_labels.shape)
        y_test = np.zeros(one_hot_labels.shape)
        y_train[train_mask, :] = one_hot_labels[train_mask, :]
        y_val[val_mask, :] = one_hot_labels[val_mask, :]
        y_test[test_mask, :] = one_hot_labels[test_mask, :]
    
        # number of samples in each set
        print('# of training samples: ', np.sum(train_mask))
        print('# of validation samples: ', np.sum(val_mask))
        print('# of testing samples: ', np.sum(test_mask))
    
        tmp_labels = [item + 1 for item in all_labels]
        train_labels = train_mask * tmp_labels
        val_labels = val_mask * tmp_labels
        test_labels = test_mask * tmp_labels
    
        # distribution of train, val and test set over classes
        train_class = [train_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('train class distribution:', train_class)
        val_class = [val_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('val class distribution:', val_class)
        test_class = [test_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('test class distribution:', test_class)
    
        # saving initial boolean masks for later use
        init_train_mask = train_mask
        init_val_mask = val_mask
        init_test_mask = test_mask
    
        # changing mask for having weighted loss
        train_mask = node_weights * train_mask
        val_mask = node_weights * val_mask
        test_mask = node_weights * test_mask
        
#        best_val_acc = 0
#        wait = 0
        # Train model
        cost_val = []
#        train_results = []
        for epoch in range(epochs):
            t = time.time()
    #
    #        # Training step
           # model.fit(graph0, y_train, sample_weight=train_mask, batch_size=557, epochs=1, shuffle=False, verbose=0)
            model.train_on_batch(graph0, y_train, sample_weight=train_mask)
            # Predict on full dataset
            preds = model.predict(graph0, batch_size=557)
        
            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                           [init_train_mask, init_val_mask])
            cost_val.append(train_val_loss[1])
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))
            
        test_loss, test_acc = evaluate_preds(preds, [y_test], [init_test_mask])
        Final_Score.append(test_acc[0])                    
        model.save('Mixed/Mix2_best_model_%d.h5'%(fold))
            
    print(Final_Score)
    print(np.mean(Final_Score)) 
#    totle_score.append(np.mean(Final_Score))  
#    print(totle_score)
#    print(np.mean(totle_score))     
         
def test():
    sparsity_threshold = 0.5
    age_adj, gender_adj, fdg_adj, apoe_adj, mixed_adj, features, all_labels, one_hot_labels, node_weights, dense_features = \
        load_tadpole_data(sparsity_threshold)
    adj_dict = {'age': age_adj, 'gender': gender_adj, 'fdg': fdg_adj, 'apoe': apoe_adj, 'mixed': mixed_adj}
    num_class = 3    
    num_folds = 10
    MAX_DEGREE = 3  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    weight_deacy = 5e-4
    ACT = 'elu'
    feature=dense_features#features[2]        
    # Normalize X
    feature = Reader.preprocess_features((sp.csr_matrix(feature, dtype=np.float32).todense()))
    print(feature.shape)
    
    adj1 = sp.csr_matrix(mixed_adj, dtype=np.float32)
    adj2 = np.load('W_G2_2.npy')
    adj2 = sp.csr_matrix(adj2/4., dtype=np.float32)
          
    print('Using Chebyshev polynomial basis filters...')
    T_k0 = Compute_Matrix(adj1) 
    T_k1 = Compute_Matrix(adj2) 
    graph0 = [feature]+T_k0+T_k1
    support = MAX_DEGREE + 1
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
    G1 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
    
    # index of fold for validation set and test set  
    val_part = 0
    test_part = 1    
    Final_Score = []
    # Num_folds cross validation
    for fold in range(num_folds):#num_folds
        print('Starting fold {}'.format(fold + 1))
          
        model = MineModel(feature, support, ACT, weight_deacy, G, G1)
        # Create model
        num_nodes = dense_features.shape[0]
        fold_size = int(num_nodes / num_folds)
        
        # shape of features
        print('whole features shape: ', dense_features.shape)
    
        # rotating folds of val and test
        val_part = (val_part + 1) % 10
        test_part = (test_part + 1) % 10
        print('fold_{},val_ind_{},test_ind_{} '.format(fold + 1, val_part, test_part))
    
        # defining train, val and test mask
        train_mask = np.ones((num_nodes,), dtype=np.bool)
        val_mask = np.zeros((num_nodes,), dtype=np.bool)
        test_mask = np.zeros((num_nodes,), dtype=np.bool)
        train_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 0
        train_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 0
        val_mask[val_part * fold_size: min((val_part + 1) * fold_size, num_nodes)] = 1
        test_mask[test_part * fold_size: min((test_part + 1) * fold_size, num_nodes)] = 1
    
        # defining train, val and test labels
        y_train = np.zeros(one_hot_labels.shape)
        y_val = np.zeros(one_hot_labels.shape)
        y_test = np.zeros(one_hot_labels.shape)
        y_train[train_mask, :] = one_hot_labels[train_mask, :]
        y_val[val_mask, :] = one_hot_labels[val_mask, :]
        y_test[test_mask, :] = one_hot_labels[test_mask, :]
    
        # number of samples in each set
        print('# of training samples: ', np.sum(train_mask))
        print('# of validation samples: ', np.sum(val_mask))
        print('# of testing samples: ', np.sum(test_mask))
    
        tmp_labels = [item + 1 for item in all_labels]
        train_labels = train_mask * tmp_labels
        val_labels = val_mask * tmp_labels
        test_labels = test_mask * tmp_labels
    
        # distribution of train, val and test set over classes
        train_class = [train_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('train class distribution:', train_class)
        val_class = [val_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('val class distribution:', val_class)
        test_class = [test_labels.tolist().count(i) for i in range(1, num_class + 1)]
        print('test class distribution:', test_class)
    
        # saving initial boolean masks for later use
        init_train_mask = train_mask
        init_val_mask = val_mask
        init_test_mask = test_mask
    
        # changing mask for having weighted loss
        train_mask = node_weights * train_mask
        val_mask = node_weights * val_mask
        test_mask = node_weights * test_mask
        
        model.load_weights('Mixed/Mix2_best_model_%d.h5'%(fold))
        preds = model.predict(graph0, batch_size=557)        
        test_loss, test_acc = evaluate_preds(preds, [y_test], [init_test_mask])
        Final_Score.append(test_acc[0]) 
    print(Final_Score)
    print(np.mean(Final_Score))
    
if __name__ == '__main__':   
    Training()      
#    test()
    
    
#    sparsity_threshold = 0.5
#    age_adj, gender_adj, fdg_adj, apoe_adj, mixed_adj, features, all_labels, one_hot_labels, node_weights, dense_features = \
#        load_tadpole_data(sparsity_threshold)
        # Early stopping
#        if epoch> 80:
#            if train_val_acc[1] >= best_val_acc:
#                
#                if wait >10:
#                    print('number:', wait)
#                    print('Epoch {}: early stopping'.format(epoch))
#                    test_loss, test_acc = evaluate_preds(preds, [y_test], [init_test_mask])
#                    Final_Score.append(test_acc[0])                    
#                    model.save('model/Time_%d_best_model_test0_%d.h5'%(num,ID))
#                    break
#                
#                elif train_val_acc[1] > best_val_acc :
#                    best_val_acc = train_val_acc[1]
#                    wait = 0
#                    print('---------------%d'%epoch)
##                #flage = model
#            else:
#                wait += 1
#                if wait >20 : #and best_val_loss < np.mean(cost_val[-(wait+1):-1])
#                    print('Epoch {}: early stopping'.format(epoch))               
#                    test_loss, test_acc = evaluate_preds(preds, [y_test], [init_test_mask])
#                    Final_Score.append(test_acc[0])
#                    model.save('model/Time_%d_best_model_test0_%d.h5'%(num,ID))
#                    print("Final_Test set results:",
#                          "loss= {:.4f}".format(test_loss[0]),
#                          "accuracy= {:.4f}".format(test_acc[0]))                
#                    break         
#
#
 
#print(np.mean((totle_score)))    
#print(totle_score)









        # Evaluation on val set


        # Evaluation on test set


        # Print results of train, val and test
#        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_results[1]),
#              "train_acc=", "{:.5f}".format(train_results[2]), "val_loss=", "{:.5f}".format(val_cost),
#              "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))
#        print("Test set results:", "test_loss=", "{:.5f}".format(test_cost),
#              "test_accuracy=", "{:.5f}".format(test_acc))
#
#        # Check val loss for early stopping
#        if epoch > max(FLAGS.early_stopping, FLAGS.start_stopping) and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#            print("Early stopping on epoch {}...".format(epoch + 1))
#            break
#
#    num_epochs.append(epoch)
#    print("Optimization Finished!")
#
#    # Collecting final results of train, test & val
#    train_accuracy.append(train_results[2])
#    val_accuracy.append(val_acc)
#    test_accuracy.append(test_acc)
#    
#
#print('Average number of epochs: {:.3f}'.format(np.mean(num_epochs)))
#print('Accuracy on {} folds'.format(num_folds))
#print('train:', train_accuracy)
#print('val', val_accuracy)
#print('test', test_accuracy)
#print()
#
# print('Test auc on {} folds'.format(num_folds))
# print(test_auc)
# print()
#
# test_avg_auc = np.mean(test_auc)
# print('Average test auc on {} folds'.format(num_folds))
# print(test_avg_auc, '±', np.std(test_auc))
#
#
