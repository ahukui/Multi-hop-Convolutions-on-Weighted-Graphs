#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:45:40 2019

@author: labadmin
"""
from __future__ import division
from __future__ import print_function
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import add,Activation
from keras import backend as K
from layers.graph import GraphConvolution
import time
from utils import *
import utils2 as Ut2
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from graph_fusing_layer import GraphFlusing
from utils import *
import utils_mine as Reader
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio


def Graph_information_Load(ID):
    connectivity = 'correlation'                # type of connectivity used for network construction
    # Get class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    
    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    
    num_classes = 2
    num_nodes = len(subject_IDs)
    
    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=np.int)
    
    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])
    
    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name='ho')

    # Compute population graph using gender and acquisition site
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)#, 'SITE_ID'
    
    
    distv = distance.pdist(features, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    graph_w = graph * sparse_graph    
    
    skf = StratifiedKFold(n_splits=10)#, shuffle= True
    
    cv_splits = list(skf.split(features, np.squeeze(y)))
    #
#    ID = 3
    train_ind = cv_splits[ID][0]
        

    test_ind = cv_splits[ID][1]
    val_ind = test_ind

    #yy = np.squeeze(y)
    labeled_ind = Reader.site_percentage(train_ind, 1, subject_IDs)
    # feature selection/dimensionality reduction step
    features = Reader.feature_selection(features, y, labeled_ind, 2000)
    
    y_train, y_val, y_test, train_mask, val_mask, test_mask = Reader.get_train_test_masks(y_data, train_ind, val_ind, test_ind)
    weight_train_mask = Reader.weight_mask(graph, train_mask, test_mask)
    print('ok')

    return graph_w, features, y_data, y_train, y_val, y_test, train_mask, val_mask, test_mask, weight_train_mask


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
    Y0 = GraphConvolution(2, support, activation=ACT)([H0]+G)

    H1 = GraphConvolution(16, support, activation=ACT, kernel_regularizer=l2(weight_deacy))([H]+G1)
    H1 = Dropout(0.3)(H1)
    Y1 = GraphConvolution(2, support, activation=ACT)([H1]+G1)

    out = GraphFlusing('attention',activation='softmax')([Y0,Y1])#Activation('softmax')(add([Y0,Y1]))#
    
    # Compile model
    model = Model(inputs=[X_in1] +G +G1, outputs=out)
    model.compile(loss='categorical_crossentropy',weighted_metrics=['acc'], optimizer=Adam(lr=0.005))
    return model
        


def Training():
    
    Final_Score = [] 
    for ID in range(1,2):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~',ID)
        A, X, y, y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = Graph_information_Load(ID)
        #y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
    #    train_mask = idx_train
        print(train_mask[idx_test])
        ## Define parameters
        MAX_DEGREE = 3  # maximum polynomial degree
        SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
        weight_deacy = 5e-4
        ACT = 'elu'
        
        # Normalize X
        X = Reader.preprocess_features((sp.csr_matrix(X, dtype=np.float32).todense()))
        print(X.shape)
        
        A0 = sp.csr_matrix(A, dtype=np.float32)
        A1 = sp.csr_matrix(np.load('new_G2.npy'), dtype=np.float32)
        
        print('Using Chebyshev polynomial basis filters...')
        T_k0 = Compute_Matrix(A0, MAX_DEGREE, SYM_NORM)
        T_k1 = Compute_Matrix(A1, MAX_DEGREE, SYM_NORM)
        graph0 = [X]+T_k0+T_k1
        print(len(graph0))
        
        support = MAX_DEGREE + 1
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
        G1 = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
        print(len(G))
        
        model = MineModel(X, support, ACT, weight_deacy, G, G1)
        
        # Helper variables for main training loop
        wait = 0
        preds = None

        best_val_acc = 0
        NB_EPOCH = 200
#        PATIENCE = 10  # early stopping patience
        
        cost_val = []
        
        for epoch in range(1, NB_EPOCH+1):
        
            # Log wall-clock time
            t = time.time()
        
            # Single training iteration (we mask nodes without labels for loss calculation)
            model.train_on_batch(graph0, y_train, sample_weight=train_mask)
        
            # Predict on full dataset
            preds = model.predict(graph0, batch_size=A.shape[0])
        
        
            # Train / validation scores
            train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                           [idx_train, idx_val])
            cost_val.append(train_val_loss[1])
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))
        
            # Early stopping
            if epoch> 80:
                if train_val_acc[1] >= best_val_acc:
                    
                    if wait >10:
                        print('number:', wait)
                        print('Epoch {}: early stopping'.format(epoch))
                        test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
                        Final_Score.append(test_acc[0])                    
                        model.save('Final_best_model_%d.h5'%(ID))
                        break
                    
                    elif train_val_acc[1] > best_val_acc :
                        best_val_acc = train_val_acc[1]
                        wait = 0
                        print('---------------%d'%epoch)
    #                #flage = model
                else:
                    wait += 1
                    if wait >20 : #and best_val_loss < np.mean(cost_val[-(wait+1):-1])
                        print('Epoch {}: early stopping'.format(epoch))               
                        test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
                        Final_Score.append(test_acc[0])
                        model.save('Final_best_model_%d.h5'%(ID))
                        print("Final_Test set results:",
                              "loss= {:.4f}".format(test_loss[0]),
                              "accuracy= {:.4f}".format(test_acc[0]))                
                        break         
    print(Final_Score)
    print(np.mean(Final_Score))        
#    totle_score.append(np.mean(Final_Score))     
#print(np.mean((totle_score)))    
#print(totle_score)    
    
    
    
if __name__ == '__main__':   
    Training()      
#    test()
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
