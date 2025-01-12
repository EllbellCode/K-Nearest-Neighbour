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

##############################################################################################
#Decision Tree Learning Classifier
##############################################################################################

def gini_score(targets):
    
    sum = 0
    
    for classification in set(targets):
        
        proportion = np.sum((targets == classification).astype(int))/len(targets)
        
        sum += proportion*(1-proportion)
        
    return sum


def partition(data_matrix, targets, feature_id, threshold):
    
    left_matrix = data_matrix[data_matrix[:,feature_id] < threshold]
    left_targets = targets[data_matrix[:,feature_id] < threshold]
    
    right_matrix = data_matrix[data_matrix[:,feature_id] >= threshold]
    right_targets = targets[data_matrix[:,feature_id] >= threshold]
    
    return right_matrix, left_matrix, right_targets, left_targets
    
    
def goodness_of_split(data_matrix, targets, feature_id, threshold):
    
    right_matrix, left_matrix, right_targets, left_targets = partition(data_matrix, targets, feature_id, threshold)

    P_L = len(left_matrix)/len(data_matrix)
    P_R = len(right_matrix)/len(data_matrix)
    
    gini_left = gini_score(left_targets)
    gini_right = gini_score(right_targets)
    
    gini = gini_score(targets)
    
    quality = gini - (P_L*gini_left) - (P_R*gini_right)
    
    return quality


def sample_feature_id_threshold(data_matrix):
    
    random_dim = np.random.randint(np.shape(data_matrix)[1])
    
    random_threshold = np.random.choice(data_matrix[:,random_dim])
    
    return random_dim, random_threshold


def select_best_split(data_matrix, targets, n_attempts):
    
    quality = -1
    
    for i in range(n_attempts):
        
        dim, threshold = sample_feature_id_threshold(data_matrix)
        
        quality_new = goodness_of_split(data_matrix, targets, dim, threshold)
        
        if quality_new > quality:
            
            quality = quality_new
            
            best_dim = dim
            
            best_threshold = threshold
            
    return best_dim, best_threshold


def build_root_node():
    
    root_node = {}
    
    root_node["splits"] = []
    root_node["terminal"] = 0
    root_node["depth"] = 0
    
    return root_node

def get_node_matrix(data_matrix, targets, input_node):
    
    node_matrix = data_matrix.copy()
    new_node_matrix = data_matrix.copy()
    node_targets = targets.copy()
    
    if len(input_node["splits"]) > 0:
        
        for i in range(len(input_node["splits"])):
            
            node_matrix = new_node_matrix.copy()
            
            dim = input_node["splits"][i][0]
            threshold = input_node["splits"][i][1]
            
            if input_node["splits"][i][2] == "L":
                
                new_node_matrix = node_matrix[node_matrix[:,dim] < threshold]
                
            else:
                
                new_node_matrix = node_matrix[node_matrix[:,dim] >= threshold]
                
               
        node_index = [np.where(new_node_matrix[i] == data_matrix)[0][0] for i in range(len(new_node_matrix))] 
        node_targets = np.array([targets[i] for i  in node_index])
    
    return new_node_matrix, node_targets

def split_node(data_matrix, targets, splt_node, n_attempts):
    
    split_node_matrix, split_node_targets = get_node_matrix(data_matrix, targets, splt_node)
    
    left_node = {}
    right_node = {}
    
    dim, threshold = select_best_split(split_node_matrix, split_node_targets, n_attempts)
    
    left_node["splits"] = splt_node["splits"] + [[dim,threshold,"L"]]
    left_node["terminal"] = 0
    left_node["depth"] = splt_node["depth"] + 1
    
    left_targets = get_node_matrix(data_matrix, targets, left_node)[1] 
    
    left_node["proportions"] = np.array([np.sum((np.array(left_targets) == i).astype(int))/len(np.array(left_targets)) for i in set(targets)])
    
    right_node["splits"] = splt_node["splits"] + [[dim,threshold,"R"]]
    right_node["terminal"] = 0
    right_node["depth"] = splt_node["depth"] + 1
    
    right_targets = get_node_matrix(data_matrix, targets, right_node)[1]
    
    right_node["proportions"] = np.array([np.sum((np.array(right_targets) == i).astype(int))/len(np.array(right_targets)) for i in set(targets)]) 
    
    return left_node, right_node


def terminal(data_matrix, targets, test_node, min_size, max_depth):
    
    terminal_node_matrix, terminal_node_targets = get_node_matrix(data_matrix, targets, test_node)
    
    if len(terminal_node_matrix) >= min_size:
        
        if test_node["depth"] <= max_depth-1:
            
            if np.var(terminal_node_targets) > 0:
                
                test_node["terminal"] = 0
                
            else:
                
                test_node["terminal"] = 1
                
        else:
            
            test_node["terminal"] = 1
            
    else:
        
        test_node["terminal"] = 1
        
        
def terminal_nodes(data_matrix, targets, n_attempts, max_depth, min_size, parent_node = build_root_node(), node_list = []):
    
    left_child, right_child = split_node(data_matrix, targets, parent_node, n_attempts)
    
    terminal(data_matrix, targets, left_child, min_size, max_depth)
    terminal(data_matrix, targets, right_child, min_size, max_depth)
    
    for child_node in [left_child, right_child]:
        
        if child_node["terminal"] == 1:
            
            node_list.append(child_node)
            
        else:
            
            node_list = terminal_nodes(data_matrix, targets, n_attempts, max_depth, min_size, child_node, node_list)
            
    return node_list

 
def fit_decision_tree(data_matrix, targets, n_attempts, max_depth, min_size):
    
    terminal_nodes_list = terminal_nodes(data_matrix, targets, n_attempts, max_depth, min_size, parent_node = build_root_node(), node_list = [])
     
    number_of_classes = len(set(targets))
    
    tree_model = [number_of_classes, terminal_nodes_list]
    
    return tree_model   
      
            
def predict_proba_decision_tree(data_matrix, model):
    
    class_number = model[0]
    
    prob_mat = np.zeros((len(data_matrix),class_number))
    
    terminal_node_list = model[1]
    
    for i in range(len(data_matrix)):
        
        instance = data_matrix[i]
        
        for j in range(len(terminal_node_list)):
            
            terminal_node = terminal_node_list[j]
            
            splits = terminal_node["splits"]
            
            count = 0
             
            for k in range(len(splits)):
                
                dim = splits[k][0]
                threshold = splits[k][1]
                direction = splits[k][2]
                
                if direction == "L":
                    
                    if instance[dim] < threshold:
                        
                        count +=1
                        
                else:
                    
                    if instance[dim] >= threshold:
                        
                        count += 1
                
            if count == len(splits):
                
                prob_mat[i] = terminal_node["proportions"]
                
                break
        
    return prob_mat


def predict_decision_tree(data_matrix, model):
    
    tree_preds = np.zeros((len(data_matrix),))
    
    prob_matrix = predict_proba_decision_tree(data_matrix, model)
    
    for i in range(len(prob_matrix)):
        
        tree_preds[i] = np.argmax(prob_matrix[i])
        
    return tree_preds

def score_decision_tree(data_matrix, targets, model):
    
    score_preds = predict_decision_tree(data_matrix, model)
    
    tree_accuracy = np.sum((score_preds == targets).astype(int))/len(score_preds)
    
    return tree_accuracy

##############################################################################################
#Cost Matrix
##############################################################################################
     
def compute_prior_probs(targets):
    
    proportions = np.array([np.sum((targets == i).astype(int))/len(targets) for i in set(targets)]) 
    
    return proportions


def prob_smoothing(mtx, eps=1e-6):
    
    mtx_eps = mtx + eps
    
    row_sums = mtx_eps.sum(axis=1)
    mtx_smooth = mtx_eps / row_sums[:, np.newaxis]
    
    return mtx_smooth


def predict_with_costs(data_matrix, model, predict_proba_func, misclassification_cost_mtx, prior_probs):
    
    model_preds = predict_proba_func(data_matrix, model)
    
    cost_preds = np.zeros(len(model_preds))
    
    for i in range(len(model_preds)):
        
        for j in range(len(model_preds.T)):
            
            class_range = list(range(len(model_preds.T)))
            
            class_range.remove(j)
            
            for k in class_range:
                
                if model_preds[i][k] == 0:
                    
                    cost_preds[i] = j
                    
                else:
                    
                    preds_ratio = model_preds[i][j]/model_preds[i][k]
                    
                    cost_ratio  = (prior_probs[k]*misclassification_cost_mtx[j,k])/(prior_probs[j]*misclassification_cost_mtx[k,j])
                    
                    if cost_ratio < preds_ratio:
                        
                        cost_preds[i] = j
                        
                    else:
                        
                        cost_preds[i] = k
               
    return cost_preds

#Checkpoint 1

X,y = make_dataset(nclasses=3, ndim=2, ndim_noninformative=2, n_instances_list=[500, 1000, 250], radius=1, thickness=.35)
plot(X,y)

X,y = make_dataset(nclasses=3, ndim=2, ndim_noninformative=2, n_instances_list=[500, 1000, 250], radius=1, thickness=.2)
plot(X,y,id_axis_1=1,id_axis_2=2)

#Checkpoint 2

X,y = make_dataset(nclasses=3, ndim=3, ndim_noninformative=2, n_instances_list=[500, 1000, 250], radius=1, thickness=.3)
params = np.arange(0,30,1)*2+1
mean_errors, std_errors = performace_estimate(X, y, k=10, n_rep=5, fit_func=fit_knn, score_func=score_knn, params=params)
plt.errorbar(params, mean_errors, std_errors);

#Checkpoint 3

def fit_decision_tree_max_depth(data_matrix, targets, param):
    return fit_decision_tree(data_matrix, targets, n_attempts=10, max_depth=param, min_size=2)

X,y = make_dataset(nclasses=3, ndim=3, ndim_noninformative=2, n_instances_list=[500, 1000, 250], radius=1, thickness=.3)
params = np.arange(2,50,10)
mean_errors, std_errors = performace_estimate(X,y,k=5, n_rep=3, fit_func=fit_decision_tree_max_depth, score_func=score_decision_tree, params=params)
plt.errorbar(params, mean_errors, std_errors);

#Checkpoint 4

X,y = make_dataset(nclasses=2, ndim=3, ndim_noninformative=2, n_instances_list=[500, 1000], radius=1, thickness=.3)
misclassification_cost_mtx = np.array([[0,1],[6,0]])
prior_probs = compute_prior_probs(y)

model = fit_decision_tree(X,y, n_attempts=10, max_depth=10, min_size=3)
preds = predict_decision_tree(X,model)
print('Confusion matrix decision tree:\n', confusion_matrix(y, preds))

preds = predict_with_costs(X, model, predict_proba_decision_tree, misclassification_cost_mtx, prior_probs)
print('Confusion matrix decision tree with user misclassification cost:\n', confusion_matrix(y, preds))

model = fit_knn(X,y,10)
preds = predict_knn(X,model)
print('Confusion matrix knn:\n', confusion_matrix(y, preds))

preds = predict_with_costs(X, model, predict_proba_knn, misclassification_cost_mtx, prior_probs)
print('Confusion matrix knn with user misclassification cost:\n', confusion_matrix(y, preds))
 

