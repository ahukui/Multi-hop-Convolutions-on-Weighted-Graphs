import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.spatial import distance
import sys
import csv
import pandas as pd
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import ABIDEParser as Reader


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def one_hot_to_class(one_hot_labels):
    num_rows, num_class = one_hot_labels.shape
    class_labels = np.zeros(shape=(num_rows,))
    for i in range(num_rows):
        class_labels[i] = np.argmax(one_hot_labels[i, :])
    return class_labels


# Dimensionality reduction on feature vectors using a ridge classifier
def feature_selection(features, labels, train_idx, num_features_selected):
    estimator = RidgeClassifier()
    selector = RFE(estimator, num_features_selected, step=100, verbose=1)
    train_features = features[train_idx, :]
    train_labels = labels[train_idx]
    selector = selector.fit(train_features, train_labels)
    new_features_selected = selector.transform(features)
    return new_features_selected


def data_generator(means, covariances, num_sample, threshold):
    num_clusters = means.shape[0]
    num_nodes = num_sample * num_clusters

    samples = np.empty((num_clusters, num_sample, means.shape[1]), dtype=float)
    labels = np.empty((num_nodes,), dtype=int)
    for i in range(num_clusters):
        samples[i, :] = np.random.multivariate_normal(mean=means[i, :], cov=covariances[i, :, :], size=(num_sample, ))
        labels[i * num_sample: (i + 1) * num_sample] = i
    features = samples.reshape((-1, means.shape[1]))

    adj = np.zeros((num_nodes,num_nodes), dtype=float)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.dot(features[i] - features[j], features[i] - features[j]) < threshold:
                adj[i, j] = adj[j, i] = 1

    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    features = features[idx, :]
    labels = [labels[item] for item in idx]
    adj = adj[idx, :]
    adj = adj[:, idx]

    one_hot_labels = np.zeros((num_nodes, num_clusters))
    one_hot_labels[np.arange(num_nodes), labels] = 1

    sparse_features = sparse_to_tuple(sp.coo_matrix(features))
    return features, sparse_features, adj, labels, one_hot_labels


def data_generator2(feature_means, feature_covariances, graph_means, graph_covariances, num_sample, threshold):
    num_clusters = feature_means.shape[0]
    num_nodes = num_sample * num_clusters
    num_graphs = graph_means.shape[0]

    samples = np.empty((num_clusters, num_sample, feature_means.shape[1]), dtype=float)
    labels = np.empty((num_nodes,), dtype=int)
    for i in range(num_clusters):
        samples[i] = np.random.multivariate_normal(mean=feature_means[i, :], cov=feature_covariances[i, :, :],
                                                   size=(num_sample,))
        labels[i * num_sample: (i + 1) * num_sample] = i
    features = samples.reshape((-1, feature_means.shape[1]))

    graph_features = np.empty((num_graphs, num_clusters, num_sample, graph_means.shape[2]), dtype=float)
    for i in range(num_graphs):
        for j in range(num_clusters):
            graph_features[i, j] = np.random.multivariate_normal(mean=graph_means[i, j, :],
                                                                 cov=graph_covariances[i, j, :, :],
                                                                 size=(num_sample,))
    graph_features = graph_features.reshape((num_graphs, -1, graph_means.shape[2]))
    print(graph_features.shape)
    adj = np.zeros((num_graphs, num_nodes, num_nodes), dtype=float)
    for i in range(num_graphs):
        for j in range(num_nodes):
            for k in range(j + 1, num_nodes):
                if np.dot(graph_features[i, j] - graph_features[i, k], graph_features[i, j] - graph_features[i, k, :]) < threshold:
                    adj[i, j, k] = adj[i, k, j] = 1

    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    features = features[idx, :]
    labels = [labels[item] for item in idx]
    adj = adj[:, idx, :]
    adj = adj[:, :, idx]

    one_hot_labels = np.zeros((num_nodes, num_clusters))
    one_hot_labels[np.arange(num_nodes), labels] = 1

    sparse_features = sparse_to_tuple(sp.coo_matrix(features))
    return features, sparse_features, adj, labels, one_hot_labels


def load_ppmi_data(sparsity_threshold):
    with open('./PPMI_dataset/idx_patients.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        ids = [int(float(row[0])) for row in rows]

    with open('./PPMI_dataset/predictionEpoch21New.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        features = np.asarray([list(map(float, row)) for row in rows])

    labels = np.zeros((len(ids),), dtype=int)
    with open('./PPMI_dataset/LABELS_PPMI.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            labels[ids.index(int(row[0]))] = int(row[1]) - 1

    features = features[1:, :]
    ids = ids[1:]
    labels = labels[1:]

    xls = pd.ExcelFile('./PPMI_dataset/Non-imagingPPMI.xls')

    updrs = pd.read_excel(xls, 'UPDRS')
    selected_columns = [column for column in updrs.columns if column[0:3] == 'NP3']
    selected_columns.append('NHY')
    selected_columns.append('PATNO')
    selected_columns.append('EVENT_ID')
    updrs = updrs[selected_columns]
    updrs = updrs.dropna(axis=0, how='any')
    # moca = pd.read_excel(xls, 'MOCA')
    # moca = moca[['PATNO', 'MCATOT', 'EVENT_ID']]

    not_BL_SC_list = []
    for patient_id in ids:
        updrs_patient_rows = updrs.loc[updrs['PATNO'] == patient_id]
        # moca_patient_rows = moca.loc[updrs['PATNO'] == patient_id]
        # if updrs_patient_rows.loc[updrs_patient_rows['EVENT_ID'] == 'BL'].empty or moca_patient_rows.loc[moca_patient_rows['EVENT_ID'] == 'SC'].empty:
        if updrs_patient_rows.loc[updrs_patient_rows['EVENT_ID'] == 'BL'].empty:
            not_BL_SC_list.append(patient_id)

    for i in not_BL_SC_list:
        idx = ids.index(i)
        features = np.delete(features, idx, axis=0)
        labels = np.delete(labels, idx, axis=0)
        ids.pop(idx)

    updrs = [np.sum(updrs.loc[updrs['PATNO'] == patient_id].iloc[0].values[:-2]) for patient_id in ids]

    gender_age = pd.read_excel(xls, 'Gender_and_Age')
    gender = gender_age[['PATNO', 'GENDER']]
    age = gender_age[['PATNO', 'BIRTHDT']]

    gender = [0 if gender.loc[gender['PATNO'] == patient_id].iloc[0].values[1] <= 1 else 1 for patient_id in ids]
    age = [2018 - age.loc[age['PATNO'] == patient_id].iloc[0].values[1] for patient_id in ids]

    num_nodes = len(ids)

    # Age affinity
    age_threshold = 2
    age_affinity = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.abs(age[i] - age[j]) <= age_threshold:
                age_affinity[i, j] = age_affinity[j, i] = 1

    # Gender affinity
    gender_affinity = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if gender[i] == gender[j]:
                gender_affinity[i, j] = gender_affinity[j, i] = 1

    # updrs affinity
    updrs_new = np.asarray(updrs).reshape((len(updrs), 1))
    column_sum = np.array(updrs_new.sum(0))
    r_inv = np.power(column_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    updrs_new = updrs_new.dot(r_mat_inv)

    dist = distance.pdist(updrs_new, metric='euclidean')
    dist = distance.squareform(dist)
    sigma = np.mean(dist)
    weights = np.exp(- dist ** 2 / (2 * sigma ** 2))

    updrs_affinity = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (updrs[i] <= 32 and updrs[j] <= 32) or (updrs[i] > 32 and updrs[j] > 32):
                updrs_affinity[i, j] = updrs_affinity[j, i] = weights[i, j]

    # features = np.asarray(features)
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
    age_affinity *= w
    gender_affinity *= w
    updrs_affinity *= w

    mixed_affinity = (age_affinity + gender_affinity + updrs_affinity) / 3

    c_1 = [i for i in range(num_nodes) if labels[i] == 0]
    c_2 = [i for i in range(num_nodes) if labels[i] == 1]

    # imbalanced
    c_1_num = len(c_1)
    c_2_num = len(c_2)
    node_weights = np.zeros((num_nodes,))
    node_weights[c_1] = 1 - c_1_num / float(num_nodes)
    node_weights[c_2] = 1 - c_2_num / float(num_nodes)

    num_labels = 2
    one_hot_labels = np.zeros((num_nodes, num_labels))
    one_hot_labels[np.arange(num_nodes), labels] = 1

    return gender_affinity, age_affinity, updrs_affinity, mixed_affinity, labels, one_hot_labels, node_weights, features


def load_ABIDE_data():
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
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')

    #gender_adj = np.zeros((num_nodes, num_nodes))
    gender_adj = Reader.create_affinity_graph_from_scores(['SEX'], subject_IDs)
    #site_adj = np.zeros((num_nodes, num_nodes))
    site_adj = Reader.create_affinity_graph_from_scores([ 'SITE_ID'], subject_IDs)
    mixed_adj = gender_adj+ site_adj

    c_1 = [i for i in range(num_nodes) if y[i] == 1]
    c_2 = [i for i in range(num_nodes) if y[i] == 2]

    # print(idx)
    y_data = np.asarray(y_data, dtype=int)
    num_labels = 2
    #one_hot_labels = np.zeros((num_nodes, num_labels))
    #one_hot_labels[np.arange(num_nodes), y_data] = 1
    sparse_features = sparse_to_tuple(sp.coo_matrix(features))
    return gender_adj, site_adj, mixed_adj, sparse_features, y ,y_data, features

def load_citation_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("citation_datasets/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("citation_datasets/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    class_labels = one_hot_to_class(labels)
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, class_labels
    return adj, features, class_labels, labels


def load_mit_data(adj_type):
    num_nodes = 84
    with open('MIT_dataset/calls.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        call_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            call_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/politics.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        politics_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            politics_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/subject_organization.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        subject_organization_adj = np.zeros((num_nodes, num_nodes))
        i = 0
        for row in rows:
            subject_organization_adj[i, :] = [float(item) for item in row]
            i += 1
    with open('MIT_dataset/cluster_labels_norm.csv') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        labels = np.zeros((num_nodes,), dtype=np.int32)
        i = 0
        for row in rows:
            labels[i] = float(row[0])
            i += 1

    features = sparse_to_tuple(sp.coo_matrix(np.eye(num_nodes)))

    if adj_type == 'calls':
        adj = call_adj
    elif adj_type == 'politics':
        adj = politics_adj
    elif adj_type == 'subject':
        adj = subject_organization_adj
    else:
        raise NotImplementedError

    idx = np.arange(num_nodes)
    np.random.shuffle(idx)
    train_proportion = 0.6
    val_proportion = 0.2
    train_idx = idx[:int(train_proportion * num_nodes)]
    val_idx = idx[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)]
    test_idx = idx[int((train_proportion + val_proportion) * num_nodes):]

    train_mask = sample_mask(train_idx, num_nodes)
    val_mask = sample_mask(val_idx, num_nodes)
    test_mask = sample_mask(test_idx, num_nodes)

    num_labels = np.max(labels)
    one_hot_labels = np.zeros((num_nodes, int(num_labels + 1)))
    one_hot_labels[np.arange(num_nodes), labels] = 1

    train_label = np.zeros(one_hot_labels.shape)
    val_label = np.zeros(one_hot_labels.shape)
    test_label = np.zeros(one_hot_labels.shape)
    train_label[train_mask, :] = one_hot_labels[train_mask, :]
    val_label[val_mask, :] = one_hot_labels[val_mask, :]
    test_label[test_mask, :] = one_hot_labels[test_mask, :]

    return adj, features, train_label, val_label, test_label, train_mask, val_mask, test_mask, labels


c1_ind=[95,37,64,127 ,156, 146,  46, 118,  32,  88, 143,  60,  27, 125,  53, 129, 110,  21,
 101,  93,  16, 115,   8, 140, 102,  28,  72, 100, 137, 155,  31, 114, 158,  29,  44 , 39,
  10, 116, 132, 133, 157, 109,  62, 144,   2,  66, 122,   3,  74,  47, 108, 111,  94 , 30,
 147, 124,  26,  63,  22, 123,  61,  13,  55,  99,  54, 136,  36,  84,  65,  23,  96 , 40,
  14,  69, 139,  98,  15, 107,  85,   5,  75,  11,  68,  49,  82, 149,  19, 138, 104 ,131,
  33,  25,  79,  77, 135, 121, 105,  71,  12,   0,  34,  70,  89,  41,   4,  97, 134 ,150,
  20,  90,  51,  42, 142,  87,  67,   7, 148,  57,  50,  38,  80,  76, 120,   1, 113 , 52,
  86, 130,  18,  92, 141, 112, 154,  45, 128,  59, 117,  48,   6, 126,  35, 151,  73,  81,
  58, 145,  83, 152,  17, 103,  43,  56,  78, 153,  24, 106,  91, 119,   9]
#
c2_ind = [93, 231, 9, 21, 64, 149, 254, 291, 221, 304, 263, 140, 165, 253, 211, 160, 132, 48, 
          268, 292, 280, 186, 216, 287, 67, 109, 31, 71, 258, 37, 227, 294, 224, 78, 158, 184, 
          204, 10, 146, 176, 119, 39, 207, 127, 261, 297, 79, 122, 203, 225, 202, 218, 96, 266, 
          58, 82, 135, 153, 59, 92, 123, 311, 17, 166, 296, 138, 85, 50, 312, 246, 233, 139, 281, 
          25, 111, 95, 275, 313, 262, 2, 161, 70, 16, 106, 47, 271, 156, 198, 205, 201, 49, 38, 
          4, 197, 101, 274, 32, 3, 267, 126, 44, 217, 53, 20, 114, 177, 257, 316, 52, 164, 30, 
          120, 110, 249, 15, 270, 236, 256, 214, 273, 174, 234, 63, 121, 240, 157, 272, 192, 0, 
          75, 22, 286, 307, 210, 7, 238, 171, 76, 74, 212, 88, 183, 187, 100, 251, 200, 73, 290, 
          317, 116, 112, 230, 241, 195, 301, 152, 209, 314, 222, 61, 191, 155, 172, 179, 151, 42, 
          83, 108, 81, 180, 300, 97, 259, 302, 141, 55, 264, 45, 36, 277, 298, 154, 315, 163, 33, 
          299, 305, 84, 56, 208, 170, 6, 150, 124, 77, 118, 228, 148, 136, 173, 282, 60, 295, 46,
          57, 131, 279, 196, 104, 133, 289, 190, 193, 232, 54, 243, 89, 99, 242, 219, 169, 113, 
          189, 162, 178, 43, 143, 80, 260, 248, 115, 34, 66, 24, 94, 91, 199, 8, 62, 252, 276, 
          147, 245, 247, 40, 269, 220, 168, 283, 285, 14, 68, 223, 175, 284, 308, 128, 98, 125, 
          194, 12, 65, 239, 265, 306, 26, 235, 117, 11, 19, 137, 144, 27, 237, 134, 29, 35, 103, 
          102, 244, 142, 86, 41, 188, 213, 1, 105, 51, 310, 250, 167, 90, 215, 309, 130, 278, 69, 
          87, 288, 185, 13, 182, 293, 28, 107, 145, 18, 159, 23, 303, 229, 255, 226, 72, 181, 206,
          129, 5]
#
c3_ind = [37, 74, 4, 69, 50, 14, 10, 22, 76, 53, 68, 64, 57, 45, 52, 12, 8, 65, 70, 18, 72, 11, 58, 
          61, 26, 59, 51, 33, 73, 77, 42, 44, 21, 54, 27, 34, 79, 66, 6, 63, 9, 46, 60, 24, 78, 5, 
          2, 32, 67, 38, 43, 36, 41, 1, 40, 30, 16, 47, 15, 20, 62, 19, 35, 7, 13, 75, 55, 71, 3, 17, 
          31, 0, 56, 23, 49, 39, 25, 28, 48, 29]

#
id_ind = [524, 285, 210, 341, 191, 168, 92, 281, 122, 124, 205, 98, 1, 490, 10, 436, 496, 277, 71, 360, 
          441, 250, 543, 115, 391, 397, 547, 195, 532, 232, 258, 276, 428, 329, 133, 252, 160, 119, 91, 
          248, 65, 352, 35, 367, 396, 555, 108, 413, 11, 353, 520, 67, 500, 186, 334, 261, 203, 442, 481, 
          178, 23, 303, 525, 126, 123, 42, 376, 5, 217, 3, 489, 128, 136, 209, 287, 30, 70, 173, 310, 112, 
          542, 235, 263, 154, 147, 529, 511, 80, 486, 221, 57, 349, 172, 134, 280, 523, 141, 517, 405, 536, 
          348, 503, 86, 6, 343, 135, 85, 58, 236, 127, 479, 132, 510, 457, 179, 275, 452, 300, 253, 495, 34, 
          468, 7, 382, 199, 471, 364, 414, 501, 316, 162, 111, 66, 372, 410, 270, 24, 220, 84, 291, 395, 439, 
          39, 82, 27, 219, 302, 100, 539, 512, 224, 139, 68, 321, 504, 394, 359, 116, 292, 249, 526, 369, 425, 
          62, 102, 327, 215, 474, 171, 319, 440, 407, 22, 230, 182, 237, 470, 314, 540, 51, 473, 59, 298, 159, 
          48, 415, 447, 332, 421, 138, 475, 18, 163, 399, 361, 223, 502, 326, 505, 521, 164, 117, 380, 330, 477, 
          469, 279, 533, 266, 16, 274, 336, 392, 363, 328, 125, 416, 278, 33, 368, 355, 204, 385, 358, 216, 97, 77, 
          238, 74, 393, 286, 72, 242, 322, 430, 515, 260, 130, 509, 389, 401, 99, 315, 63, 507, 482, 158, 131, 29, 
          513, 295, 150, 460, 273, 243, 81, 412, 55, 554, 480, 381, 438, 129, 69, 60, 454, 420, 268, 366, 426, 356, 
          388, 41, 247, 228, 229, 15, 551, 357, 386, 109, 448, 105, 152, 20, 214, 251, 375, 296, 347, 293, 79, 143, 
          183, 21, 184, 174, 121, 190, 2, 402, 535, 350, 225, 202, 373, 419, 36, 101, 218, 244, 54, 148, 255, 272, 
          107, 96, 294, 26, 553, 188, 444, 307, 95, 32, 257, 196, 193, 181, 254, 46, 549, 4, 506, 478, 331, 404, 61, 
          354, 156, 289, 301, 38, 403, 137, 12, 282, 449, 463, 313, 176, 290, 340, 518, 508, 318, 431, 64, 445, 552, 
          87, 466, 493, 239, 312, 451, 103, 75, 9, 231, 544, 213, 240, 462, 25, 387, 140, 259, 43, 149, 379, 398, 446, 
          528, 56, 104, 13, 550, 233, 464, 498, 185, 49, 212, 378, 409, 390, 383, 351, 333, 317, 201, 484, 151, 262, 339, 
          189, 207, 73, 265, 146, 522, 423, 206, 465, 200, 297, 406, 530, 94, 177, 155, 308, 443, 422, 345, 226, 45, 245, 
          17, 455, 40, 417, 114, 344, 534, 264, 435, 374, 418, 161, 19, 271, 305, 78, 456, 541, 467, 499, 170, 432, 113, 8, 
          145, 384, 198, 437, 427, 556, 90, 365, 342, 299, 411, 485, 494, 157, 256, 0, 514, 50, 335, 28, 44, 208, 323, 548, 
          76, 346, 267, 89, 142, 338, 371, 306, 429, 370, 377, 227, 144, 546, 246, 284, 37, 450, 362, 31, 47, 192, 488, 110, 
          545, 538, 241, 476, 453, 53, 519, 165, 458, 187, 83, 461, 434, 106, 194, 516, 472, 433, 324, 169, 311, 492, 320, 
          487, 288, 175, 491, 180, 88, 269, 93, 222, 325, 400, 459, 483, 309, 304, 166, 120, 424, 408, 118, 211, 167, 153, 497, 
          283, 537, 14, 52, 337, 531, 197, 234, 527]




def Random_shuffle(x, indx):
    y = []
    for i in indx:
        y.append(x[i])
    return y
    


def load_tadpole_data(sparsity_threshold):
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
        apoe_affinity *= w
        age_affinity *= w
        gender_affinity *= w
        fdg_affinity *= w
#        print(apoe_affinity)
        mixed_affinity = (age_affinity + gender_affinity + fdg_affinity + apoe_affinity) / 4
#        print(labels)
        c_1 = [i for i in range(num_nodes) if labels[i] == 0]
        c_2 = [i for i in range(num_nodes) if labels[i] == 1]
        c_3 = [i for i in range(num_nodes) if labels[i] == 2]

        # imbalanced
        c_1_num = len(c_1)
        c_2_num = len(c_2)
        c_3_num = len(c_3)
        num_nodes = c_1_num + c_2_num + c_3_num
#        print(c_1_num,c_2_num,c_3_num,num_nodes)
        c_1 = Random_shuffle(c_1, c1_ind)
        c_2 = Random_shuffle(c_2, c2_ind)
        c_3 = Random_shuffle(c_3, c3_ind)            
            
        selection_c_1 = c_1[:c_1_num]
#        print(len(c_1),len(selection_c_1))
        selection_c_2 = c_2[:c_2_num]
        selection_c_3 = c_3[:c_3_num]
        idx = np.concatenate((selection_c_1, selection_c_2, selection_c_3), axis=0)

        node_weights = np.zeros((num_nodes,))
        node_weights[selection_c_1] = 1 - c_1_num / float(num_nodes)
        node_weights[selection_c_2] = 1 - c_2_num / float(num_nodes)
        node_weights[selection_c_3] = 1 - c_3_num / float(num_nodes)
        idx = Random_shuffle(idx, id_ind)
        
#        np.random.shuffle(idx)
#        xxid = np.random.permutation(len(idx)) 


        
        features = features[idx, :]
        labels = [labels[item] for item in idx]

        age_affinity = age_affinity[idx, :]
        age_affinity = age_affinity[:, idx]

        gender_affinity = gender_affinity[idx, :]
        gender_affinity = gender_affinity[:, idx]

        fdg_affinity = fdg_affinity[idx, :]
        fdg_affinity = fdg_affinity[:, idx]

        apoe_affinity = apoe_affinity[idx, :]
        apoe_affinity = apoe_affinity[:, idx]
        print(apoe_affinity)
        # adj = adj[idx, :]
        # adj = adj[:, idx]
        node_weights = node_weights[idx]





        # plot features
        # pca = PCA(n_components=5)
        # pca.fit_transform(features)
        # transformed = pca.transform(features)
        # print(pca.explained_variance_)
        # print(pca.components_)
        # print(pca.mean_)
        # components = [0, 3]
        # plt.scatter(transformed[:, components[0]], transformed[:, components[1]], c=labels,
        #             cmap=plt.cm.get_cmap('spectral', 3))
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')
        # plt.colorbar()
        # plt.show()

        # train_proportion = 0.8
        # val_proportion = 0.1

        # train_mask = np.zeros((num_nodes,), dtype=np.bool)
        # val_mask = np.zeros((num_nodes,), dtype=np.bool)
        # test_mask = np.zeros((num_nodes,), dtype=np.bool)
        # train_mask[:int(train_proportion * num_nodes)] = 1
        # val_mask[int(train_proportion * num_nodes): int((train_proportion + val_proportion) * num_nodes)] = 1
        # test_mask[int((train_proportion + val_proportion) * num_nodes):] = 1

        num_labels = 3
        one_hot_labels = np.zeros((num_nodes, num_labels))
        one_hot_labels[np.arange(num_nodes), labels] = 1

        # train_label = np.zeros(one_hot_labels.shape)
        # val_label = np.zeros(one_hot_labels.shape)
        # test_label = np.zeros(one_hot_labels.shape)
        # train_label[train_mask, :] = one_hot_labels[train_mask,:]
        # val_label[val_mask, :] = one_hot_labels[val_mask, :]
        # test_label[test_mask, :] = one_hot_labels[test_mask, :]

        # train_mask = node_weights * train_mask
        # val_mask = node_weights * val_mask
        # test_mask = node_weights * test_mask
        # SVM performance
        # train_features = features[train_idx, :]
        # train_labels = [labels[i] for i in train_idx]
        # test_features = features[test_idx, :]
        # test_labels = [labels[i] for i in test_idx]
        # svc2 = svm.SVC(kernel='linear').fit(train_features, train_labels)
        # train_pred = svc2.predict(train_features)
        # test_pred = svc2.predict(test_features)
        # print('test acc:', np.mean(np.equal(test_pred, test_labels)))
        # print('train acc:', np.mean(np.equal(train_pred, train_labels)))
        sparse_features = sparse_to_tuple(sp.coo_matrix(features))

        # return adj, features, train_label, val_label, test_label, train_mask, val_mask, test_mask, labels
        return age_affinity, gender_affinity, fdg_affinity, apoe_affinity, mixed_affinity, sparse_features, labels, one_hot_labels, node_weights, features


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))



def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


