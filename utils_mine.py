from __future__ import print_function

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import csv
import scipy.io as sio
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome

# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

# Input data variables
root_folder = '/home/labadmin/GCN/data/'
data_folder = os.path.join(root_folder, 'Outputs/cpac/filt_noglobal/func_preproc/')
phenotype = os.path.join(root_folder, 'Phenotypic_V1_0b_preprocessed1.csv')


def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))
#        print(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')
#            print(subject_IDs[i])

    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """
    data_folder = '/home/labadmin/GCN/data/Outputs/cpac/filt_noglobal/rois_ho/'
    timeseries = []
#    ro_file = [f for f in os.listdir(data_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
#    print(ro_file)
    subject_folder = data_folder #os.path.join(data_folder, subject_list[i])    
    for i in range(len(subject_list)):
        
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith(str(subject_list[i])+'_rois_' + atlas_name + '.1D')]
#        print(ro_file)
        fl = os.path.join(subject_folder, ro_file[0])
#        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """
    data_folder = './'
    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
#    print('------------',featureX.shape, featureY.shape)
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    print(x_data.shape)
    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    data_folder ='/home/labadmin/GCN/ABIDE/population-gcn-master/matfile/'
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)
#    print('!!!!!!!!!!!----------',type(matrix))
    return matrix


# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def weight_mask(graph, train_mask, test_mask):
    weight_train_mask = np.zeros_like(train_mask,dtype='float64')
    weight_train_mask[train_mask ==  True] = 1
    weight_train_mask[train_mask ==  False] = 0 
#    mask = weight_train_mask
#    print(weight_train_mask)
    
    for i in range(test_mask.shape[0]):
        if test_mask[i]:
            total_num = np.sum(graph[i,:])
#            print(total_num)
            for col in range(graph.shape[1]):
                if graph[i, col]:
                    weight_train_mask[col] += graph[i, col]/total_num
#                    print(weight_train_mask[col])
    train_mask = train_mask.astype('float64')
    weight_train_mask *= train_mask
    return weight_train_mask
                    



def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def cheack_file():
    import glob
    subject_IDs = get_ids()
#    print(subject_IDs[10:15])
#    xx = fetch_filenames(subject_IDs, 'func_preproc')    
    atlas_name = 'ho'
    xx = get_timeseries(subject_IDs, atlas_name)
    print(xx[10].shape)
    
#    xx = fetch_filenames(subject_IDs, 'func_preproc')
#    print(xx)
#    xx = os.listdir(data_folder+'func_preproc/')
#    for file_name in xx:
#        for i in range(len(subject_IDs)):
#            if  glob.glob('*' + subject_IDs[i] + '_func_preproc.nii.gz')

#    xx = fetch_filenames(subject_IDs, 'func_preproc')
#    print(xx)
#        try:
#            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])

def New_downLoad():
    import urllib.request
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/'\
                'ABIDE_Initiative'
    s3_path = '/'.join([s3_prefix, 'Outputs/cpac/filt_noglobal/func_preproc/'])

    print(s3_path)
    download_dir = '/home/labadmin/GCN/data/Outputs/'
    
#    s3_pheno_file = urllib.request(s3_path)
#    print(s3_pheno_file)
#    pheno_list = s3_pheno_file.read()
#    print((pheno_list))

    import glob
    import glob
    subject_IDs = get_ids()
    file_type = 'func_preproc'
    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = 'CMU_a_0050665_func_preproc.nii.gz'
    urllib.request.urlretrieve(s3_path+filenames,download_dir+filenames)
    # Fill list with requested file paths
#    for i in range(len(subject_IDs)):
#        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))
##        print(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
#        try:
#            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
#        except IndexError:
#            # Return N/A if subject ID is not found
#            print('--------------------',subject_IDs[i])
#            strid = 2
#            file_name = glob.glob('*' + str(int(subject_IDs[i])-strid) + filemapping[file_type])
#            if file_nam

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


def load_data(dataset_str):
    """Load data."""
    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, 'data/')

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
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

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

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
    return (features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return (adj_normalized)



if __name__ == "__main__":

    Graph = np.asarray([[0,1,2,0],
                        [1,0,0,1],
                        [2,0,0,1],
                        [0,1,1,0]])
    
    print(Graph.shape)
    
    
    train_mask = np.asarray([True,  False, False, True])
    test_mask = np.asarray([False, True, True,  False])  
    print(train_mask)
    xx = weight_mask(Graph, train_mask, test_mask)
    print(xx)





#def normalize_adj(adj, symmetric=True):
#    if symmetric:
#        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
#        a_norm = adj.dot(d).transpose().dot(d).tocsr()
#    else:
#        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
#        a_norm = d.dot(adj).tocsr()
#    return a_norm
#
#
#def preprocess_adj(adj, symmetric=True):
#    adj = adj + sp.eye(adj.shape[0])
#    adj = normalize_adj(adj, symmetric)
#    return adj



