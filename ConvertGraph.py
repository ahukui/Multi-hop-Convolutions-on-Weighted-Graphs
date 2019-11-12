#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:58:09 2019

@author: labadmin
"""
import csv
import numpy as np
from sklearn import svm
def sort_based_on_weight(list_item,list_weigth):
#    old_weigth = copy.deepcopy(list_weigth)
    new_list = []    
    if len(list_weigth):
        for i in range(len(list_weigth)):
            idex = np.argmax(list_weigth)
#            print(idex)
            list_weigth[idex] = -10000
            new_list.append(list_item[idex])
        return new_list
    else:
        return list_item            
#        new_weigth = sorted(list_weigth,reverse = True)
#    #    print(new_weigth, list_weigth)
#        index_list =[]
#        for item in new_weigth:
#            index_list.append(list_weigth.index(item))
#    #    print(index_list)
#        
#        print()
#        new_list = []
#        for i in index_list:
#            if list_item[i] not in new_list:
#                new_list.append(list_item[i])
#            
#        return new_list
#    else:
#        return list_item
def shuffle_train(data, mask):
    length=data.shape[0]
    perm = np.random.permutation(length)
    print(perm)    
    print(type(perm))

    data = data[perm]
    mask = mask[perm]
    return data, mask    



def Get_new_graph(graph):
    row,col = graph.shape
    new_graph2 = np.zeros_like(graph)
    new_graph3 = np.zeros_like(graph)    
    new_graph4 = np.zeros_like(graph)    
    new_graph5 = np.zeros_like(graph)    
    new_graph6 = np.zeros_like(graph)
    ########first order

    for row_node in range(row):
        print('----',row_node)
        adj = []
#        weight = []
#        adj.append(row_node)
        for col_node in range(col):
            if graph[row_node,col_node] and col_node not in adj:
                adj.append(col_node)
#                weight.append(graph[row_node,col_node])
#        print(adj)
#        print(adj)        
#        adj = sort_based_on_weight(adj,weight)            
#        print(adj)                
        if len(adj): 
            adj0 = []
            weight = []
            for f_node2 in adj:
#                print(f_node2)
                for col_node in range(col):
                    if graph[f_node2,col_node] and col_node not in adj and col_node != row_node:
                        if graph[row_node,f_node2] + graph[f_node2,col_node] > new_graph2[row_node,col_node]:
                            new_graph2[row_node,col_node] = graph[row_node,f_node2] + graph[f_node2,col_node]
                        adj0.append(col_node)
                        weight.append(graph[f_node2,col_node])

            adj.append(row_node)
#            adj0 = sort_based_on_weight(adj0,weight)   
                                  
            if len(adj0): 
                adj1 = []
#                weight = []
                for f_node3 in adj0:
                    for col_node in range(col):
                        if graph[f_node3,col_node] and col_node not in adj and col_node not in adj0:
                            if new_graph2[row_node,f_node3] + graph[f_node3,col_node] > new_graph3[row_node,col_node]:
                               new_graph3[row_node,col_node] = new_graph2[row_node,f_node3] + graph[f_node3,col_node]
                            adj1.append(col_node) 
#                            weight.append(graph[f_node3,col_node])
    
#                adj1 = sort_based_on_weight(adj1,weight) 
                if len(adj1): 
                    adj2 = []
#                    weight = []
                    for f_node4 in adj1:
                        for col_node in range(col):
                            if graph[f_node4,col_node] and col_node not in adj and col_node not in adj0 and col_node not in adj1:
                                if new_graph4[row_node,col_node] < (new_graph3[row_node,f_node4] + graph[f_node4,col_node]):
                                   new_graph4[row_node,col_node] = new_graph3[row_node,f_node4] + graph[f_node4,col_node]
                                adj2.append(col_node)
#                                weight.append(graph[f_node4,col_node])
        
#                    adj2 = sort_based_on_weight(adj2,weight) 
#                    if len(adj2): 
#                        adj3 = []
##                        weight = []
#                        for f_node5 in adj2:
#                            for col_node in range(col):
#                                if graph[f_node5,col_node] and col_node not in adj and col_node not in adj0 and col_node not in adj1 and col_node not in adj2:
#                                    if new_graph5[row_node,col_node] < (new_graph4[row_node,f_node5] + graph[f_node5,col_node]):
#                                        new_graph5[row_node,col_node] = new_graph4[row_node,f_node5] + graph[f_node5,col_node]
#                                    adj3.append(col_node)
##                                    weight.append(graph[f_node5,col_node])
#                        
##                        adj3 = sort_based_on_weight(adj3,weight) 
#                        if len(adj3): 
#                            adj4 = []
##                            weight = []
#                            for f_node6 in adj3:
#                                for col_node in range(col):
#                                    if graph[f_node6,col_node] and col_node not in adj and col_node not in adj0 and col_node not in adj1 and col_node not in adj2 and col_node not in adj3:
#                                        if new_graph6[row_node,col_node] < (new_graph5[row_node,f_node6] + graph[f_node6,col_node]):
#                                            new_graph6[row_node,col_node] = new_graph5[row_node,f_node6] + graph[f_node6,col_node]
#                                        adj4.append(col_node)
#                                        weight.append(graph[f_node6,col_node])


                        
#    if skip==2:
#         return new_graph2/skip
#                        
#    elif skip==3:
#         return new_graph3/skip   
#                     
#    elif skip==4:
#         return new_graph4/skip
#                        
#    elif skip==5:
#         return new_graph5/skip
#    else:
#        return new_graph6/skip
    return new_graph2/2, new_graph3/3, new_graph4/4#, new_graph5/5, new_graph6/6


def Get_new_graph_V2(graph):
    row,col = graph.shape
    Index_list = [x for x in range(col)]
    new_graph2 = np.zeros_like(graph)
    new_graph3 = np.zeros_like(graph)    
    new_graph4 = np.zeros_like(graph)    
#    new_graph5 = np.zeros_like(graph)    [[0 0 0 4 0 4 0 0 0]
#    new_graph6 = np.zeros_like(graph)
    ########first order
     
    for row_node in range(row):
        print('----',row_node)
        adj = []

        adj.append(row_node)
        for col_node in range(col):
            if graph[row_node,col_node] and col_node not in adj:
                adj.append(col_node)
        
        
        if len(adj): 
            adj0 = []
            for f_node2 in adj:
#                print(f_node2)
                for col_node in list(set(Index_list) - set(adj)):
                    if graph[f_node2,col_node]:
                        if graph[row_node,f_node2] + graph[f_node2,col_node] > new_graph2[row_node,col_node]:
                            new_graph2[row_node,col_node] = graph[row_node,f_node2] + graph[f_node2,col_node]
                        adj0.append(col_node)
                       
            if len(adj0): 
                adj1 = []
#                weight = []
                for f_node3 in adj0:
                    for col_node in list(set(Index_list) - set(adj)-set(adj0)):
                        if graph[f_node3,col_node]:
                            if new_graph2[row_node,f_node3] + graph[f_node3,col_node] > new_graph3[row_node,col_node]:
                               new_graph3[row_node,col_node] = new_graph2[row_node,f_node3] + graph[f_node3,col_node]
                            adj1.append(col_node) 
#            
    

                if len(adj1): 
                    adj2 = []

                    for f_node4 in adj1:
                        for col_node in list(set(Index_list) - set(adj)-set(adj0)-set(adj1)):
                            if graph[f_node4,col_node]:
                                if new_graph4[row_node,col_node] < (new_graph3[row_node,f_node4] + graph[f_node4,col_node]):
                                   new_graph4[row_node,col_node] = new_graph3[row_node,f_node4] + graph[f_node4,col_node]
                                adj2.append(col_node)
#                                
        
#                    adj2 = sort_based_on_weight(adj2,weight) 
#                    if len(adj2): 
#                        adj3 = []
##                        weight = []
#                        for f_node5 in adj2:
#                            for col_node in range(col):
#                                if graph[f_node5,col_node] and col_node not in adj and col_node not in adj0 and col_node not in adj1 and col_node not in adj2:
#                                    if new_graph5[row_node,col_node] < (new_graph4[row_node,f_node5] + graph[f_node5,col_node]):
#                                        new_graph5[row_node,col_node] = new_graph4[row_node,f_node5] + graph[f_node5,col_node]
#                                    adj3.append(col_node)
##                                    weight.append(graph[f_node5,col_node])
#                        
##                        adj3 = sort_based_on_weight(adj3,weight) 
#                        if len(adj3): 
#                            adj4 = []
##                            weight = []
#                            for f_node6 in adj3:
#                                for col_node in range(col):
#                                    if graph[f_node6,col_node] and col_node not in adj and col_node not in adj0 and col_node not in adj1 and col_node not in adj2 and col_node not in adj3:
#                                        if new_graph6[row_node,col_node] < (new_graph5[row_node,f_node6] + graph[f_node6,col_node]):
#                                            new_graph6[row_node,col_node] = new_graph5[row_node,f_node6] + graph[f_node6,col_node]
#                                        adj4.append(col_node)
#                                        weight.append(graph[f_node6,col_node])


#        return new_graph6/skip
    return new_graph2, new_graph3, new_graph4#, new_graph5/5, new_graph6/6

def Get_new_graph_V3(graph):
    row,col = graph.shape
    Index_list = [x for x in range(col)]
    new_graph2 = np.zeros_like(graph)
    new_graph3 = np.zeros_like(graph)  
    new_graph4 = np.zeros_like(graph)     
    ########first order
    for row_node in range(row):
        print('----',row_node)
        adj = []
        adj.append(row_node)
        for col_node in range(col):
            if graph[row_node,col_node] and col_node not in adj:
                adj.append(col_node)
        
        if len(adj): 
            adj0 = []
            for f_node2 in adj:
#                print(f_node2)
                for col_node in list(set(Index_list) - set(adj)):
                    if graph[f_node2,col_node]:
                        if graph[row_node,f_node2] + graph[f_node2,col_node] > new_graph2[row_node,col_node]:
                            new_graph2[row_node,col_node] = graph[row_node,f_node2] + graph[f_node2,col_node]
                        adj0.append(col_node)
                       
            if len(adj0): 
                adj1 = []
#                weight = []
                for f_node3 in adj0:
                    for col_node in list(set(Index_list) - set(adj)-set(adj0)):
                        if graph[f_node3,col_node]:
                            if new_graph2[row_node,f_node3] + graph[f_node3,col_node] > new_graph3[row_node,col_node]:
                               new_graph3[row_node,col_node] = new_graph2[row_node,f_node3] + graph[f_node3,col_node]
                            adj1.append(col_node) 


                if len(adj1): 
                    adj2 = []

                    for f_node4 in adj1:
                        for col_node in list(set(Index_list) - set(adj)-set(adj0)-set(adj1)):
                            if graph[f_node4,col_node]:
                                if new_graph4[row_node,col_node] < (new_graph3[row_node,f_node4] + graph[f_node4,col_node]):
                                   new_graph4[row_node,col_node] = new_graph3[row_node,f_node4] + graph[f_node4,col_node]
                                adj2.append(col_node)
                            
#            
    return new_graph2, new_graph3, new_graph4

def Get_tadpole_graph(sparsity_threshold = 0.5):
    with open('tadpole_dataset/tadpole_2.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        apoe = []
        ages = []
        gender = []
        fdg = []
        features = []
        labels = []
        cnt = 0
        apoe_col_num = 0
        age_col_num = 0
        gender_col_num = 0
        fdg_col_num = 0
        label_col_num = 0
        for row in rows:
            if cnt != 0:
                row_features = row[fdg_col_num + 1:]
                if row_features.count('') == 0 and row[apoe_col_num] != '':
                    apoe.append(int(row[apoe_col_num]))
                    ages.append(float(row[age_col_num]))
                    gender.append(row[gender_col_num])
                    fdg.append(float(row[fdg_col_num]))
                    labels.append(int(row[label_col_num]) - 1)
                    features.append([float(item) for item in row_features])
                    cnt += 1
            else:
                apoe_col_num = row.index('APOE4')
                age_col_num = row.index('AGE')
                gender_col_num = row.index('PTGENDER')
                fdg_col_num = row.index('FDG')
                label_col_num = row.index('DXCHANGE')
                cnt += 1

        num_nodes = len(labels)

        apoe_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if apoe[i] == apoe[j]:
                    apoe_affinity[i, j] = apoe_affinity[j, i] = 1

        age_threshold = 2
        age_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.abs(ages[i] - ages[j]) <= age_threshold:
                    age_affinity[i, j] = age_affinity[j, i] = 1

        gender_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if gender[i] == gender[j]:
                    gender_affinity[i, j] = gender_affinity[j, i] = 1

        reshaped_fdg = np.reshape(np.asarray(fdg), newshape=[-1, 1])
        svc = svm.SVC(kernel='linear').fit(reshaped_fdg, labels)
        prediction = svc.predict(reshaped_fdg)
        fdg_affinity = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if prediction[i] == prediction[j]:
                    fdg_affinity[i, j] = fdg_affinity[j, i] = 1

        features = np.asarray(features)
        column_sum = np.array(features.sum(0))
        r_inv = np.power(column_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        features = features.dot(r_mat_inv)

        dist = distance.pdist(features, metric='euclidean')
        dist = distance.squareform(dist)
        sigma = np.mean(dist)
        w = np.exp(- dist ** 2 / (2 * sigma ** 2))
        w[w < sparsity_threshold] = 0
#        apoe_affinity *= w
#        age_affinity *= w
#        gender_affinity *= w
#        fdg_affinity *= w

        mixed_affinity = (age_affinity + gender_affinity + fdg_affinity + apoe_affinity) / 4
        return mixed_affinity, w


if __name__ == '__main__': 
    
    graph, weight = Get_tadpole_graph()
#    graph[graph<1] = 0
#    G2, G3, G4 = Get_new_graph_V3(graph)
#    W_G2 = G2 * weight
#    W_G3 = G3 * weight
#    W_G4 = G4 * weight    
#    np.save('W_G2_4.npy', W_G2)
#    np.save('W_G3_4.npy', W_G3)         
    xx = np.load('W_G4_4.npy')        
    
    
    
    
    
    
    
    
    
    
    
#    graph, weight = Get_graph_weight()
##    graph -= np.clip(graph,0,1)
##    
##    G2, G3 = Get_new_graph_V3(graph)
##    W_G2 = G2 * weight
##    W_G3 = G3 * weight
##    np.save('G2_clip.npy', W_G2)
##    np.save('G3_clip.npy', W_G3)    
#    xx = np.load('new_G3.npy')
#    yy = xx/weight
#    print(np.sum(yy))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    from keras import backend as K
#    with tf.Session('') as sess:
#        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
#        A = np.asarray([[1.,2.,3.],[1.,2.,3.]])
#        print(A.shape)
#        A = tf.convert_to_tensor(A)#K.concatenate(A)
##        A.append(tf.convert_to_tensor(np.asarray([[1.0,2.0,3.0]])))
##        A.append(tf.convert_to_tensor(np.asarray([[1.0,2.0,3.0]]))) 
##        A = K.concatenate(A)
#        print('atten_map------------',K.int_shape(A))
#        #A = tf.constant(A)
#        B = K.softmax(A,0)
#        print(B.eval())
    
    
#   Index_list = [x for x in range(10)]
#   print(Index_list)      
#   a = [1,2,3]
#   Index_list = list(set(Index_list) - set(a))
#   print(Index_list)     
    
#    with tf.Session('') as sess:
#        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
##        feed_dict = {image: im0}
#        
#        Graph =tf.constant([[0,2,0,0,0,0,0,0,0],
#                               [1,0,1,0,0,0,0,0,0],
#                               [0,1,0,1,0,1,0,0,0],
#                               [0,0,1,0,1,0,0,0,0],
#                               [0,0,0,1,0,0,0,0,0],
#                               [0,0,1,0,0,0,1,1,0],
#                               [0,0,0,0,0,1,0,0,0],
#                               [0,0,0,0,0,1,0,0,1],
#                               [0,0,0,0,0,0,0,1,0]
#                               ])        
#        
#        
##        a = tf.constant(Graph)
#        print('-----',Graph.get_shape().as_list())
#        print()
#        img  =  Get_new_graph_tf(Graph)
        
    

#    Graph = np.asanyarray([[0,2,0,0,0,0,0,0,0],
#                           [2,0,1,0,0,0,0,0,0],
#                           [0,1,0,1,0,1,0,0,0],
#                           [0,0,1,0,1,0,0,0,0],
#                           [0,0,0,1,0,0,0,0,0],
#                           [0,0,1,0,0,0,1,1,0],
#                           [0,0,0,0,0,1,0,0,0],
#                           [0,0,0,0,0,1,0,0,1],
#                           [0,0,0,0,0,0,0,1,0]
#                           ])
##        
##    Graph = np.asanyarray([[0,1,2,0],
##                           [1,0,0,1],
##                           [2,0,0,1],
##                           [0,1,1,0]])    
#        
#        
#    print(Graph)
#    new2,new3,new4 = Get_new_graph_V2(Graph)
#    print(new2)
#    print(new3)        
#    print(new4)
    
#li = [1,2,3,4]
#w = [1,1,1,3]
##w.sort(reverse = True)
##print(w)
##a,b = shuffle_train(np.asarray(li),np.asarray(w))
#a = sort_based_on_weight(li,w)   
#print(a)
    
    
#
#new = Get_new_graph(Graph,3)
#print(new)
#
#new = Get_new_graph(Graph,4)
#print(new)
#
#new = Get_new_graph(Graph,5)
#print(new)
#
#
#new = Get_new_graph(Graph,6)
#print(new)