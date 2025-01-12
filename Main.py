import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mode


##############################################################################################
#Create artificial dataset to test classification algorithms
##############################################################################################



def rescale_within_shell(x, radius, thickness):
    
    magnitude = sum(np.square(x))**0.5
    normalised_x = x/magnitude
    lower_bound = normalised_x * (radius-thickness)
    upper_bound = normalised_x * (radius+thickness)
    proportion = np.random.rand()
    rescaled_x = lower_bound*proportion + upper_bound*(1-proportion)
    
    return rescaled_x
    
def make_shell(centre, radius, thickness, n_instances):
    
    df_shell = np.zeros([n_instances,len(centre)])

    for i in range(n_instances):
        
        random_vector = np.random.rand(len(centre))-0.5
        random_vector_scaled = rescale_within_shell(random_vector, radius, thickness)
        centred_vector = random_vector_scaled + centre
        df_shell[i] = centred_vector
        
    return df_shell

def make_dataset(nclasses, ndim, ndim_noninformative, n_instances_list, radius, thickness):
    
    df_data = np.zeros([sum(n_instances_list),ndim+ndim_noninformative])
    df_targets = np.zeros(sum(n_instances_list))
    
    instance = 0

    for i in range(nclasses):
        
        class_centre = np.array([i]*ndim)
            
        class_shell  = make_shell(class_centre, radius, thickness, n_instances_list[i])
        
        gaussian = np.random.randn(n_instances_list[i],ndim_noninformative)
        
        class_instances = np.hstack((class_shell,gaussian))
        
        for j in range(len(class_instances)):
            
            df_data[j+instance] = class_instances[j]
            df_targets[j+instance] = i
            
        instance += len(class_instances)
            
    return df_data,df_targets


def plot(X, y, id_axis_1=0,id_axis_2=1):
    
    plt.scatter(X[:,id_axis_1],X[:,id_axis_2], c=y, cmap = "bwr_r", edgecolor="black")
    plt.grid()

##############################################################################################
#Partition dataset into folds according to cross validation procedure
##############################################################################################
    
def shuffle(X,y):
    
    permutation = np.random.permutation(len(y))
    
    X_shuffle = X[permutation]
    y_shuffle = y[permutation]
    
    return X_shuffle, y_shuffle


def make_train_test_ids_for_fold_i(n_elements, k, fold_i):
     
    index = np.array([x for x in range(n_elements)])
    
    folds = np.array_split(index,k)
            
    test = folds[fold_i]
    
    del folds[fold_i]
    
    train = np.concatenate(folds, axis=0)
    
    return train,test


def make_cv(X,y, k = 5):
    
    dfs_train = []
    dfs_test = []
    vectors_train = []
    vectors_test = []
    
    n_classes = len(set(y))
    
    for i in range(k):
        
        class_train_dfs = []
        class_train_vectors = []
        
        class_test_dfs = []
        class_test_vectors = []
        
        index_sort = np.argsort(y)
        sorted_y = y[index_sort]
        index_start = np.unique(sorted_y, return_counts=True, return_index=True)[1]
        class_index = np.split(index_sort, index_start[1:])
        
        
        for j in range(n_classes):
            
            X_class = X[class_index[j]]
            y_class = y[class_index[j]]
            
            X_class, y_class = shuffle(X_class,y_class)
            
            train_j, test_j = make_train_test_ids_for_fold_i(len(X_class),k,0)
            
            class_train_df = X_class[train_j]
            class_train_vector = y_class[train_j]
            
            class_test_df = X_class[test_j]
            class_test_vector = y_class[test_j]
            
            class_train_dfs.append(class_train_df)
            class_train_vectors.append(class_train_vector)
        
            class_test_dfs.append(class_test_df)
            class_test_vectors.append(class_test_vector)
            
        
        df_train = np.concatenate( class_train_dfs, axis=0 )
        vector_train = np.concatenate( class_train_vectors, axis=0 )
         
        df_test = np.concatenate( class_test_dfs, axis=0 )
        vector_test = np.concatenate( class_test_vectors, axis=0 )
        
        dfs_train.append(df_train)
        vectors_train.append(vector_train)
    
        dfs_test.append(df_test)
        vectors_test.append(vector_test)
    
    return dfs_train, vectors_train, dfs_test, vectors_test

##############################################################################################
#Compute the predictive performance of classifiers
##############################################################################################

def confusion_matrix(targets, preds):
    
    confusion_matrix = np.zeros((len(set(preds)),len(set(targets))))
    
    for i in range(len(preds)):
        
        confusion_matrix[int(preds[i]), int(targets[i])] += 1
    
    return confusion_matrix

def performace_estimate(X, y, k, n_rep, fit_func, score_func, params):
    
    error_means = []
    
    error_stds = []
    
    for i in range(len(params)):
        
        value_errors = []
        
        for j in range(n_rep):
            
            dfs_train_j, vectors_train_j, dfs_test_j, vectors_test_j = make_cv(X,y,k)
                      
            for l in range(k):
                
                fit_model = fit_func(dfs_train_j[l],vectors_train_j[l], params[i])
            
                error_l = 1-score_func(dfs_test_j[l], vectors_test_j[l], fit_model)
                
                value_errors.append(error_l)
                 
        error_means.append(np.mean(value_errors))
        error_stds.append(np.std(value_errors))
        
    return np.array(error_means), np.array(error_stds)


def fit_knn(data_matrix, targets, param):
    
    knn_model = [data_matrix, targets, param]
    
    return knn_model


def predict_proba_knn(data_matrix, model):
    
    train_matrix, train_classes, param = model
    
    distances = cdist(data_matrix, train_matrix) 
    indices = distances.argsort(axis=1)
    knn_indices = indices[:, :param]                
    
    knn_classes = train_classes[knn_indices]     
    
    knn_probs = np.zeros((len(knn_classes), len(set(train_classes))))
    
    for i in range(len(knn_classes)):
        
        probs = []
        
        for classification in set(train_classes):
            
            probs.append((np.count_nonzero(knn_classes[i] == classification))/len(knn_classes[i]))
            
        knn_probs[i] = probs
        
    return knn_probs


##############################################################################################
#Implement k-nearest neighbour classifier algorithm
##############################################################################################

def predict_knn(data_matrix, model):
    
    train_matrix, train_classes, param = model
    
    distances = cdist(data_matrix, train_matrix) 
    indices = distances.argsort(axis=1)
    knn_indices = indices[:, :param]                
    knn_classes = train_classes[knn_indices] 
    
    class_modes = mode(knn_classes, axis=1)[0].flatten()
    
    return class_modes

def score_knn(data_matrix, targets, model):
    
    knn_preds = predict_knn(data_matrix, model)
    
    accuracy = np.sum((knn_preds == targets).astype(int))/len(knn_preds)
    
    return accuracy
