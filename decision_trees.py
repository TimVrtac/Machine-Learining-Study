from tkinter.ttk import Notebook
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import my_statistics as ms
from tqdm.notebook import tqdm

def plot_tree(heritage):
    #TODO: imrove node positioning -> they should not overlap
    points = {}
    lines = np.zeros((len(heritage),4))
    n_cl = len(heritage)
    for i,j in enumerate(heritage):
        if i == 0:
            points[j] = [0,0]
        else:
            parent_node = points[heritage[j]['parent'][0]]
            if int(j[1:]) % 2 == 0:
                #(0.2*i)
                points[j] = [parent_node[0] - (n_cl**-(0.015*i)), parent_node[1] + 1]
            else:
                points[j] = [parent_node[0] + (n_cl**-(0.015*i)), parent_node[1] + 1]
            lines[i,:] = parent_node[0], points[j][0], parent_node[1], points[j][1]
            
    plt.figure(figsize=(8,8))
    points_arr = np.array(list(points.values())).squeeze()
    points_arr[:,1] *= -1
    annot = list(points.keys())
    plt.scatter(points_arr[:,0], points_arr[:,1])
    for i in range(points_arr.shape[0]):
        plt.annotate(f'{annot[i]}', (points_arr[i,0], points_arr[i,1]+.2))
        plt.plot(lines[i,[0,1]], lines[i,[2,3]]*-1, 'k')
    plt.grid()

def find_s(x_in, y_ref, s):
    """
    Function determines 's' parameter for indicidual predictor by minimization of RSS.
    """
    RSS_min = np.array([np.inf])
    s_min = min(s)
    for i in s:
        R1_ind = np.argwhere(x_in<i)
        R2_ind = np.argwhere(x_in>=i)
        if (len(R1_ind) !=0) & (len(R2_ind)!= 0):
            R1_mean = np.average(y_ref[R1_ind])
            R2_mean = np.average(y_ref[R2_ind])
            R1_RSS = sum((x_in[R1_ind]-R1_mean)**2)
            R2_RSS = sum((x_in[R2_ind]-R2_mean)**2)
            RSS_s = R1_RSS + R2_RSS
            if RSS_s<RSS_min:
                RSS_min = RSS_s
                s_min = i
        else:
            pass
    return RSS_min, s_min



def split(x_in, y_ref, s_partitions):
    """
    Function finds split parameters for individual split according to recursice binary splitting method.
    x_in: pandas dataframe of input predictors
    y_ref: pandas series of reference responses
    s_partitions: number of partition between min and max predicor values when searching for minimum s value for individual predictor.
    
    return: index of predictor with min RSS, s_value for predictor with min RSS.
    """
    pred_list = list(x_in.columns)
    ref = np.array(y_ref)
    RSS_min = []
    s_min = []
    for i in pred_list:
        d_series = np.array(x_in[i])
        s_values = np.linspace(min(d_series), max(d_series),s_partitions)[1:-1] # s values for individual predictor
        RSS_, s_ = find_s(d_series, ref, s_values)
        RSS_min.append(RSS_)
        s_min.append(s_)
    min_ind = np.argmin(RSS_min)
    pred_min = pred_list[min_ind]
    s_min = s_min[min_ind]
    return pred_min, s_min


def initial(x_in):
    predictors = list(x_in.columns)
    R_dict = {}
    for i in predictors:
        R_dict[i] = [min(x_in[i]),max(x_in[i])]
    return R_dict


def decision_tree(x_in, y_ref, s_partitions, R_size_max, plot_ = False):
    R_list = {'R0': initial(x_in)}
    R_x_data = {'R0':x_in}
    R_x_temp = R_x_data.copy()
    R_y_data = {'R0':y_ref}
    R_size=np.inf
    heritage_template = {'parent':[], 'kids':[], 'level':0}
    heritage = {'R0':copy.deepcopy(heritage_template)}
    name_ = 0
    while R_size > R_size_max:
        sizes=[]
        for j, i in enumerate(R_x_temp):
            #print(j,i)
            if R_x_data[i].shape[0]>R_size_max:
                pred_, s_ = split(R_x_data[i], R_y_data[i], s_partitions=s_partitions)
                old = R_list[i][pred_].copy()
                name_+=1
                R_list[f'R{name_}'] = R_list[i].copy()
                R_list[f'R{name_}'][pred_] = [s_, old[1]]
                heritage[i]['kids'].append(f'R{name_}')
                heritage[f'R{name_}'] = copy.deepcopy(heritage_template)
                heritage[f'R{name_}']['parent'].append(i)
                heritage[f'R{name_}']['level'] = heritage[i]['level']+1
                mask_1 = list(((R_x_data[i][pred_]>=R_list[f'R{name_}'][pred_][0]) & (R_x_data[i][pred_]<=R_list[f'R{name_}'][pred_][1])))
                R_x_data[f'R{name_}'] = R_x_data[i][mask_1]
                R_y_data[f'R{name_}'] = R_y_data[i][mask_1]
                size_0 = R_x_data[f'R{name_}'].shape[0]
                if size_0 == 0:
                    del R_x_data[f'R{name_}']
                    del R_y_data[f'R{name_}']
                else:
                    sizes.append(size_0)
                name_+=1
                R_list[f'R{name_}'] = R_list[i].copy()
                R_list[f'R{name_}'][pred_] = [old[0],s_]
                heritage[f'R{name_}'] = copy.deepcopy(heritage_template)
                heritage[i]['kids'].append(f'R{name_}')
                heritage[f'R{name_}']['parent'].append(i)
                heritage[f'R{name_}']['level'] = heritage[i]['level']+1
                mask_2 = list(((R_x_data[i][pred_]>=R_list[f'R{name_}'][pred_][0]) & (R_x_data[i][pred_]<R_list[f'R{name_}'][pred_][1])))
                R_x_data[f'R{name_}'] = R_x_data[i][mask_2]
                R_y_data[f'R{name_}'] = R_y_data[i][mask_2]
                size_1 = R_x_data[f'R{name_}'].shape[0]
                if size_1 ==0:
                    del R_x_data[f'R{name_}']
                    del R_y_data[f'R{name_}']
                else:
                    sizes.append(size_1)
                del R_list[i]
                del R_x_data[i]
                del R_y_data[i]
            else:
                sizes.append(R_x_data[i].shape[0])
                pass
        R_x_temp = R_x_data.copy()
        R_size=max(sizes)
        #print(list(R_list.keys()), sizes, sum(sizes))
        #print('-----------------------------------------')
    #print(R_list)
    Y_outputs = {}
    for j in R_list:
        Y_outputs[j] = np.average(R_y_data[j])
        
    if plot_:
        plot_tree(heritage)
    # TODO: add tree pruninig inside this function to lower the number of returns
    return R_list, Y_outputs, R_y_data, heritage


def predict(input_, R_list, y_ref):
    """
    R_list - tree -> dictionary with class boundaries.
    """
    # dict for input classification
    Y_pred = np.zeros(input_.shape[0])
    Y_cl = np.zeros(input_.shape[0], dtype=object)
    predictors = list(input_.columns)
    for i in R_list:
        mask = input_.copy()
        for j in predictors:
            mask[j] = (input_[j] >= R_list[i][j][0]) & (input_[j] <= R_list[i][j][1])
        row_ind = np.argwhere(np.array(mask.all(axis=1))==True)
        Y_pred[row_ind] = y_ref[i]
        Y_cl[row_ind] = i
    return Y_pred, Y_cl


def join_leaves(y_classified, y_ref, pair, parent, return_tree=False, tree=None):
    """
    join 2 leaves into parent. pair is list of form: ['Ri1,Ri2'], parent is parent node 'Ri'
    """
    y_cl = copy.deepcopy(y_classified)
    y_r = copy.deepcopy(y_ref)
    y_cl[parent] = list(y_classified[pair[0]]) + list(y_classified[pair[1]])
    y_r[parent] = np.average(np.array(y_cl[parent]))
    del y_cl[pair[0]]
    del y_cl[pair[1]]
    del y_r[pair[0]]
    del y_r[pair[1]]
    if return_tree:
        tree_ = copy.deepcopy(tree)
        tree_[parent] = {}
        for i in tree_[pair[0]]:
            tree_[parent][i] = [min(tree_[pair[0]][i][0], tree_[pair[1]][i][0]),
                                max(tree_[pair[0]][i][1], tree_[pair[1]][i][1])]
        del tree_[pair[0]]
        del tree_[pair[1]]
        return y_cl, y_r, tree_
    
    return y_cl, y_r


def tree_score(y_classified, y_averages, alpha):
    """Function calculates the tree score for the purpose of the decision tre pruning.

    Args:
        y_classified ([type]): [description]
        y_averages ([type]): [description]
        alpha ([type]): [description]

    Returns:
        [type]: [description]
    """
    no_of_leaves = len(y_classified)
    SSR = 0
    for i in list(y_classified.keys()):
        y_ = np.array(y_classified[i])
        y_r = y_averages[i]
        SSR += sum((y_ - y_r)**2)
    penalty = alpha * no_of_leaves
    return SSR + penalty


def get_pairs(nodes):
    """
    Finds nodes with the same parents.
    """
    pairs = {}
    parents = np.unique(nodes[:,1])
    for j in parents:
        sp = []
        for i in nodes:    
            if i[1] == j:
                sp.append(i[0])
            else:
                pass
        pairs[j] = sp
    return pairs


def find_lvl_nodes(heritage, max_lvl):
    """Funtion finds nodes which belog to the entered level in the decision tree.

    Args:
        heritage ([type]): [description]
        max_lvl ([type]): [description]

    Returns:
        [type]: [description]
    """
    nodes = []
    for i in heritage:
        if heritage[i]['level'] == max_lvl:
            nodes.append(np.array((i, heritage[i]['parent'][0]), dtype=object))
        else:
            pass
    return np.array(nodes)  


def get_highest_lvl(heritage):
    """Function finds the highest level of nodes in the decision tree

    Args:
        heritage ([type]): [description]

    Returns:
        [type]: [description]
    """
    max_lvl = 0
    for i in heritage:
        i_lvl = heritage[i]['level']
        if i_lvl>max_lvl:
            max_lvl = i_lvl
        else:
            pass
    return max_lvl  


def classify(y_pred, y_cl):
    """Function determines which leaf individual prediction belong to.

    Args:
        y_pred ([type]): [description]
        y_cl ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_dict = {}
    y_cl2 = y_cl[np.nonzero(y_cl)]
    classes = np.unique(y_cl2)
    for i in classes:
        y_dict[i] = y_pred[y_cl==i]
    return y_dict


def tree_pruning(y_classified, y_averages, heritage, alpha, all_=False):
    """Implementation of tree pruning.

    Args:
        y_classified ([type]): [description]
        y_averages ([type]): [description]
        heritage ([type]): [description]
        alpha ([type]): [description]
        all_ = if True tree scores for all the nodes are returned

    Returns:
        [type]: [description]
    """
    tree_scores = []
    tree_nodes = []
    # list of node levels from highest to lowest
    max_lvl = get_highest_lvl(heritage)
    levels = np.arange(1, max_lvl+1)[::-1]
    y_cl_temp = copy.deepcopy(y_classified)
    y_avg_temp = copy.deepcopy(y_averages)
    # full tree score
    tree_scores.append(tree_score(y_classified, y_averages, alpha))
    tree_nodes.append(len(y_classified))
    for i in levels:
        lvl_nodes = find_lvl_nodes(heritage, i)
        lvl_pairs = get_pairs(lvl_nodes)
        for par in lvl_pairs:
            pair_ = lvl_pairs[par]
            # new tree
            if (pair_[0] in y_cl_temp) and (pair_[1] in y_cl_temp):
                y_cl_temp, y_avg_temp = join_leaves(y_cl_temp, y_avg_temp, pair_, par)
            else:
                pass
            tree_scores.append(tree_score(y_cl_temp, y_avg_temp, alpha))
            tree_nodes.append(len(y_cl_temp))
    min_ind = np.argmin(tree_scores)
    if all_:
        return tree_scores, tree_nodes
    return tree_scores[min_ind], tree_nodes[min_ind]


def find_alphas(y_classified, y_ref_vals, heritage, alpha_range):
    tree_sizes = []
    alpha_list = []
    tree_size=np.inf
    for i in tqdm(alpha_range):
        min_sc, size_sc = tree_pruning(y_classified, y_ref_vals, heritage, i)
        if size_sc < tree_size:
            tree_sizes.append(size_sc)
            alpha_list.append(i)
            tree_size=size_sc
    return alpha_list, tree_sizes


def shrink_tree(tree, y_classified, y_averages, heritage, size_goal):
     # list of node levels from highest to lowest
    max_lvl = get_highest_lvl(heritage)
    levels = np.arange(1, max_lvl+1)[::-1]
    y_cl_temp = copy.deepcopy(y_classified)
    y_avg_temp = copy.deepcopy(y_averages)
    new_tree = copy.deepcopy(tree)
    size = len(tree)
    for i in levels:
        lvl_nodes = find_lvl_nodes(heritage, i)
        lvl_pairs = get_pairs(lvl_nodes)
        for par in lvl_pairs:
            pair_ = lvl_pairs[par]
            # new tree
            if (pair_[0] in y_cl_temp) and (pair_[1] in y_cl_temp) and (size > size_goal):
                y_cl_temp, y_avg_temp, new_tree = join_leaves(y_cl_temp,
                                                                 y_avg_temp,
                                                                 pair_,
                                                                 par,
                                                                 return_tree=True,
                                                                 tree= new_tree)
                size = len(new_tree)
            else:
                pass
    return new_tree, y_avg_temp


np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
features = ['Hits', 'Runs', 'Walks', 'Years', 'RBI']
hitters = pd.read_csv('C:/Users/timvr/Documents/Doktorat/Introduction to statistical Learning/Gradivo/Hitters.csv')
hitters.dropna(axis=0, subset=['Salary'], inplace=True)
hitters_num = hitters.loc[:,hitters.dtypes == 'int64']
decision_tree(hitters_num[features], hitters.Salary, s_partitions=30, R_size_max=3)
