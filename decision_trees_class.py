import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import my_statistics as ms
from tqdm.notebook import tqdm
import math
import random


class DecisionTree:
    def __init__(self, training_inputs, training_outputs, s_div, criterion, max_leaf_size):
        """
        DecisionTree class' init function

        Args:
            training_inputs: training predictor values (pandas dataframe)
            training_outputs: training output data (padnas dataframe)
            s_div: number of partition between min and max predictor values when searching for minimum s value for
                   individual predictor.
            criterion: string with self.criterion for data partitioning: RSS (residual sum of squares - regression),
                                                                         Gini (Gini index - classification),
                                                                         entropy (classification))
            max_leaf_size: max size of leaves R
        """
        np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        self.s_div = s_div
        self.criterion = criterion
        self.max_leaf_size = max_leaf_size
        self.tree, self.y_ref, self.y_data_cl, self.heritage = self.generate_tree(self.training_inputs,
                                                                                  self.training_outputs)
        self.original_tree = self.tree
        self.original_y_ref = self.y_ref
        self.original_y_data_cl = self.y_data_cl
        self.original_heritage = self.heritage
        self.bagging_model = {}
        self.bagging_ref = {}
        self.random_f_model = {}
        self.random_f_ref = {}
        self.B = None

    def generate_tree(self, training_inputs_, training_outputs_, m=None):
        """Function generates decision tree on the basis od input data

        Returns:
            dict: tree: dictionary of predictor borders for individual leaves'
            dict: Y_outputs: dictionary of mean values for individual leaves'
            dict: R_y_data: dictionary of input reference data arranged according to belonging leaves
            dict: heritage: dictionary of parents and kids for each node
        """
        tree = {'R0': self.initial()}
        R_x_data = {'R0': training_inputs_}
        R_x_temp = copy.deepcopy(R_x_data)
        R_y_data = {'R0': training_outputs_}
        R_size = np.inf
        heritage_template = {'parent': [], 'kids': [], 'level': 0}
        heritage = {'R0': copy.deepcopy(heritage_template)}
        name_ = 0
        while R_size > self.max_leaf_size:
            sizes = []
            for j, i in enumerate(R_x_temp):
                if R_x_data[i].shape[0] > self.max_leaf_size:
                    pred_, s_ = self.split(R_x_data[i], R_y_data[i], m=m)
                    old = copy.deepcopy(tree[i][pred_])
                    name_ += 1
                    tree[f'R{name_}'] = copy.deepcopy(tree[i])
                    tree[f'R{name_}'][pred_] = [s_, old[1]]
                    heritage[i]['kids'].append(f'R{name_}')
                    heritage[f'R{name_}'] = copy.deepcopy(heritage_template)
                    heritage[f'R{name_}']['parent'].append(i)
                    heritage[f'R{name_}']['level'] = heritage[i]['level'] + 1
                    mask_1 = list(((R_x_data[i][pred_] >= tree[f'R{name_}'][pred_][0]) &
                                   (R_x_data[i][pred_] <= tree[f'R{name_}'][pred_][1])))
                    R_x_data[f'R{name_}'] = R_x_data[i][mask_1]
                    R_y_data[f'R{name_}'] = R_y_data[i][mask_1]
                    size_0 = R_x_data[f'R{name_}'].shape[0]
                    if size_0 == 0:
                        del R_x_data[f'R{name_}']
                        del R_y_data[f'R{name_}']
                    else:
                        sizes.append(size_0)
                    name_ += 1
                    tree[f'R{name_}'] = copy.deepcopy(tree[i])
                    tree[f'R{name_}'][pred_] = [old[0], s_]
                    heritage[i]['kids'].append(f'R{name_}')
                    heritage[f'R{name_}'] = copy.deepcopy(heritage_template)
                    heritage[f'R{name_}']['parent'].append(i)
                    heritage[f'R{name_}']['level'] = heritage[i]['level'] + 1
                    mask_2 = list(((R_x_data[i][pred_] >= tree[f'R{name_}'][pred_][0]) & (
                            R_x_data[i][pred_] < tree[f'R{name_}'][pred_][1])))
                    R_x_data[f'R{name_}'] = R_x_data[i][mask_2]
                    R_y_data[f'R{name_}'] = R_y_data[i][mask_2]
                    size_1 = R_x_data[f'R{name_}'].shape[0]
                    if size_1 == 0:
                        del tree[f'R{name_}']
                        del R_x_data[f'R{name_}']
                        del R_y_data[f'R{name_}']
                    else:
                        sizes.append(size_1)
                    del tree[i]
                    del R_x_data[i]
                    del R_y_data[i]
                else:
                    sizes.append(R_x_data[i].shape[0])

            R_x_temp = R_x_data.copy()
            R_size = max(sizes)
            # print(sizes)
        Y_outputs = {}
        for j in tree:
            if self.criterion == 'RSS':
                Y_outputs[j] = np.average(R_y_data[j])
            else:
                out = np.unique(R_y_data[j], return_counts=True)
                m_ind = np.argmax(out[1])
                Y_outputs[j] = out[0][m_ind]
        return tree, Y_outputs, R_y_data, heritage

    def initial(self):
        predictors = list(self.training_inputs.columns)
        R_dict = {}
        for i in predictors:
            R_dict[i] = [min(self.training_inputs[i]), max(self.training_inputs[i])]
        return R_dict

    def split(self, x_node, y_node, m=None):
        """
        Function finds split parameters for individual split according to recursive binary splitting method.

        Args:
            x_node: pandas dataframe of input predictors for individual node
            y_node: pandas series of reference responses for individual node
            m: number of predictors taken into account in individual split
        
        return: index of predictor with min RSS, s_value for predictor with min RSS.
        """
        pred_list = list(x_node.columns)
        if m is None:
            pass
        else:
            pred_l_temp = pred_list.copy()
            pred_m = []
            for i in range(m):
                pred_m.append(random.choice(pred_l_temp))
                pred_l_temp.remove(pred_m[i])
            pred_list = pred_m
        ref = np.array(y_node)
        err_min = []
        s_min = []
        for i in pred_list:
            d_series = np.array(x_node[i])
            s_values = np.linspace(min(d_series), max(d_series), self.s_div)[1:-1]  # s values for individual predictor
            err_, s_ = self.find_s(d_series, ref, s_values)
            err_min.append(err_)
            s_min.append(s_)
        min_ind = np.argmin(np.array(err_min).ravel())
        pred_min = pred_list[min_ind]
        s_min = s_min[min_ind]
        return pred_min, s_min

    def find_s(self, x_node, y_node, s_values):
        """
        Function determines 's' parameter for individual predictor by minimization of error.

        Args:
            x_node: pandas dataframe of input predictors for individual node
            y_node: pandas series of reference responses for individual node
            s_values: numpy array of possible 's' values

        return: min error for inserted s array, corresponding s
        """
        if self.criterion == 'RSS':
            err_min = np.array([np.inf])
        elif self.criterion.lower() == 'gini' or self.criterion.lower() == 'entropy':
            err_min = np.inf
        else:
            print('Invalid criterion')
            return
        s_min = min(s_values)
        factor = 1
        for i in s_values:
            R1_ind = np.argwhere(x_node < i)
            R2_ind = np.argwhere(x_node >= i)
            if (len(R1_ind) != 0) & (len(R2_ind) != 0):
                if self.criterion == 'RSS':
                    R1_mean = np.average(y_node[R1_ind])
                    R2_mean = np.average(y_node[R2_ind])
                    R1_RSS = ms.RSS(x_node[R1_ind], R1_mean)
                    R2_RSS = ms.RSS(x_node[R2_ind], R2_mean)
                    err_s = R1_RSS + R2_RSS
                elif (self.criterion.lower() == 'gini') or (self.criterion == 'entropy'):
                    y_R1 = y_node[R1_ind].flatten()
                    y1_unique = np.unique(y_R1)
                    GD1 = 0
                    for k in y1_unique:
                        i_ind = np.argwhere(y_R1 == k).flatten()
                        pm_i = len(y_R1[i_ind]) / len(y_R1)
                        if self.criterion.lower() == 'gini':
                            factor = 1
                            GD1 += np.array(pm_i * (1 - pm_i))
                        else:
                            # due to the logarithm final result has to be multiplied py -1 in order to obtain positive
                            # error
                            factor = -1
                            GD1 += np.array(pm_i * np.log(pm_i))
                    y_R2 = y_node[R2_ind].flatten()
                    y2_unique = np.unique(y_R2)
                    GD2 = 0
                    for j in y2_unique:
                        j_ind = np.argwhere(y_R2 == j).flatten()
                        pm_j = len(y_R2[j_ind]) / len(y_R2)
                        if self.criterion.lower() == 'gini':
                            GD2 += np.array(pm_j * (1 - pm_j))
                            factor = 1
                        else:
                            GD2 += np.array(pm_j * np.log(pm_j))
                            factor = -1
                    err_s = GD1 * factor + GD2 * factor
                else:
                    return

                if err_s < err_min:
                    err_min = err_s
                    s_min = i
            else:
                pass
        return err_min, s_min

    def plot_tree(self):
        """
        Function plots generated decision tree.
        """
        # TODO: improve node positioning -> they should not overlap
        # TODO: correct heritage when shrinking the tree!
        points = {}
        lines = np.zeros((len(self.heritage), 4))
        n_cl = len(self.heritage)
        for i, j in enumerate(self.heritage):
            if i == 0:
                points[j] = [0, 0]
            else:
                parent_node = points[self.heritage[j]['parent'][0]]
                if int(j[1:]) % 2 == 0:
                    # (0.2*i)
                    points[j] = [parent_node[0] - (n_cl ** -(0.015 * i)), parent_node[1] + 1]
                else:
                    points[j] = [parent_node[0] + (n_cl ** -(0.015 * i)), parent_node[1] + 1]
                lines[i, :] = parent_node[0], points[j][0], parent_node[1], points[j][1]

        plt.figure(figsize=(8, 8))
        points_arr = np.array(list(points.values())).squeeze()
        points_arr[:, 1] *= -1
        annot = list(points.keys())
        plt.scatter(points_arr[:, 0], points_arr[:, 1])
        for i in range(points_arr.shape[0]):
            plt.annotate(f'{annot[i]}', (points_arr[i, 0], points_arr[i, 1] + .2))
            plt.plot(lines[i, [0, 1]], lines[i, [2, 3]] * -1, 'k')
        plt.grid()

    def predict(self, input_):
        """
        Function makes predictions for input predictor values.
        Args:
            input_: input predictor values

        Returns: numeric array of predictions, dictionary of form {class:[output values]}

        """
        # dict for input classification
        Y_pred = np.zeros(input_.shape[0])
        Y_cl = np.zeros(input_.shape[0], dtype=object)
        predictors = list(input_.columns)
        for i in self.tree:
            mask = input_.copy()
            for j in predictors:
                mask[j] = (input_[j] >= self.tree[i][j][0]) & (input_[j] <= self.tree[i][j][1])
            row_ind = np.argwhere(np.array(mask.all(axis=1)) == True)
            Y_pred[row_ind] = self.y_ref[i]
            Y_cl[row_ind] = i
        return Y_pred, Y_cl

    def tree_pruning(self, alpha, all_=False):
        """
        Implementation of tree pruning.
        Args:
            alpha: tree complexity penalty parameter
            all_: if True tree scores for all the nodes are returned

        Returns: min tree score, tree size for minimum tree score.

        """
        """

        Args:
            alpha (float): 
            all_: 

        Returns:
            [type]: [description]
        """
        tree_scores = []
        tree_nodes = []
        # list of node levels from highest to lowest
        max_lvl = self.get_highest_lvl()
        levels = np.arange(1, max_lvl + 1)[::-1]
        y_cl_temp = copy.deepcopy(self.y_data_cl)
        y_avg_temp = copy.deepcopy(self.y_ref)
        # full tree score
        tree_scores.append(self.tree_score(self.y_data_cl, self.y_ref, alpha))
        tree_nodes.append(len(self.y_data_cl))
        for i in levels:
            lvl_nodes = self.find_lvl_nodes(i)
            lvl_pairs = self.get_pairs(nodes=lvl_nodes)
            for par in lvl_pairs:
                pair_ = lvl_pairs[par]
                # new tree
                if (pair_[0] in y_cl_temp) and (pair_[1] in y_cl_temp):
                    y_cl_temp, y_avg_temp = self.join_leaves(y_cl_temp, y_avg_temp, pair_, par)
                    tree_scores.append(self.tree_score(y_cl_temp, y_avg_temp, alpha))
                    tree_nodes.append(len(y_cl_temp))
                else:
                    pass
        min_ind = np.argmin(tree_scores)
        if all_:
            return tree_scores, tree_nodes
        return tree_scores[min_ind], tree_nodes[min_ind]

    def join_leaves(self, y_classified, y_ref, pair, parent, return_tree=False, tree=None):
        """
        Function joins 2 leaves into parent. pair is , parent is parent node 'Ri'
        Args:
            y_classified: dictionary of form {class:[output values]}
            y_ref: dictionary of form {class:corresponding mean output value}
            pair: list of form: ['Ri1,Ri2']
            parent: node 'Ri'
            return_tree:
            tree:

        Returns: updated y_classified, y_ref dictionaries

        """
        """
        
        """
        y_cl = copy.deepcopy(y_classified)
        y_r = copy.deepcopy(y_ref)
        y_cl[parent] = list(y_classified[pair[0]]) + list(y_classified[pair[1]])
        if self.criterion == 'RSS':
            y_r[parent] = np.average(np.array(y_cl[parent]))
        else:
            out = np.unique(y_cl[parent], return_counts=True)
            m_ind = np.argmax(out[1])
            y_r[parent] = out[0][m_ind]
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

    @staticmethod
    def tree_score(y_classified, y_averages, alpha):
        """
        Function calculates the tree score for the purpose of the decision tree pruning.
        Args:
            y_classified: dictionary of form {class:[output values]}
            y_averages: dictionary of form {class:corresponding mean output value}
            alpha: tree complexity penalty parameter

        Returns: tree score value

        """
        no_of_leaves = len(y_classified)
        SSR = 0
        for i in list(y_classified.keys()):
            y_ = np.array(y_classified[i])
            y_r = y_averages[i]
            SSR += sum((y_ - y_r) ** 2)
        penalty = alpha * no_of_leaves
        return SSR + penalty

    @staticmethod
    def get_pairs(nodes):
        """
        Finds nodes with the same parents.
        Args:
            nodes: numeric array with columns (nodes, parents)

        Returns:

        """
        """
        
        """
        pairs = {}
        parents = np.unique(nodes[:, 1])
        for j in parents:
            sp = []
            for i in nodes:
                if i[1] == j:
                    sp.append(i[0])
                else:
                    pass
            pairs[j] = sp
        return pairs

    def find_lvl_nodes(self, lvl):
        """
        Function finds nodes which belong to the entered level in the decision tree.
        Args:
            lvl: level of nodes in tree hierarchy we would like to find

        Returns: list of nodes belonging to the level of interest
        """
        nodes = []
        for i in self.heritage:
            if self.heritage[i]['level'] == lvl:
                nodes.append(np.array((i, self.heritage[i]['parent'][0]), dtype=object))
            else:
                pass
        return np.array(nodes)

    def get_highest_lvl(self):
        """Function finds the highest level of nodes in the decision tree

        Returns:
            int: highest node level in the tree hierarchy
        """
        max_lvl = 0
        for i in self.heritage:
            i_lvl = self.heritage[i]['level']
            if i_lvl > max_lvl:
                max_lvl = i_lvl
            else:
                pass
        return max_lvl

    @staticmethod
    def classify(y_pred, y_cl):
        """
        Function determines which leaf individual prediction belong to.
        Args:
            y_pred: numeric array of predicted output values (numpy array)
            y_cl: numeric array of classes to which belong individual predicted vaules (numpy array)

        Returns: dictionary {class:np.array([predictions])}

        """
        y_dict = {}
        y_cl2 = y_cl[np.nonzero(y_cl)]
        classes = np.unique(y_cl2)
        for i in classes:
            y_dict[i] = y_pred[y_cl == i]
        return y_dict

    def find_alphas(self, alpha_range):
        """
        Find alpha values which correspond to different tree sizes.
        Args:
            alpha_range: Numeric array of considered alpha values (numpy array)

        Returns: list of alphas, list of corresponding tree sizes

        """
        tree_sizes = []
        alpha_list = []
        tree_size = np.inf
        for i in tqdm(alpha_range):
            _, size_sc = self.tree_pruning(alpha=i)
            if size_sc < tree_size:
                tree_sizes.append(size_sc)
                alpha_list.append(i)
                tree_size = size_sc
        return alpha_list, tree_sizes

    # tree size
    def shrink_tree(self, size_goal):
        """
        Function shrinks tree to the desired size by joining nodes from top to the bottom of the tree.
        Args:
            size_goal: desired tree size (int)

        Returns: None

        """
        # list of node levels from highest to lowest
        max_lvl = self.get_highest_lvl()
        levels = np.arange(1, max_lvl + 1)[::-1]
        y_cl_temp = copy.deepcopy(self.y_data_cl)
        y_avg_temp = copy.deepcopy(self.y_ref)
        new_tree = copy.deepcopy(self.tree)
        size = len(self.tree)
        for i in levels:
            lvl_nodes = self.find_lvl_nodes(i)
            lvl_pairs = self.get_pairs(lvl_nodes)
            for par in lvl_pairs:
                pair_ = lvl_pairs[par]
                # new tree
                if (pair_[0] in y_cl_temp) and (pair_[1] in y_cl_temp) and (size > size_goal):
                    y_cl_temp, y_avg_temp, new_tree = self.join_leaves(y_cl_temp,
                                                                       y_avg_temp,
                                                                       pair_,
                                                                       par,
                                                                       return_tree=True,
                                                                       tree=new_tree)
                    size = len(new_tree)
                else:
                    pass
        self.tree = new_tree
        self.y_ref = y_avg_temp

    def to_original(self):
        self.tree = self.original_tree
        self.y_data_cl = self.original_y_data_cl
        self.y_ref = self.original_y_ref

    # bagging
    def bagging(self, B):
        """
        Function generates series of decision trees using bootstrapped training data
        Returns: list of decision trees
        """
        self.B = B
        self.bagging_model = {}
        self.bagging_ref = {}
        bootstrap_ind = self.bootstrapping(B=B)

        for i in tqdm(range(B)):
            # print(i)
            self.bagging_model[i], self.bagging_ref[i], _, _ = self.generate_tree(
                self.training_inputs.iloc[bootstrap_ind[i, :], :],
                self.training_outputs[bootstrap_ind[i, :]])

    def predict_bag_rf(self, input_, model, ref_):
        """
                Function makes predictions for input predictor values.
                Args:
                    input_: input predictor value
                    model: model obtained from bagging or random forest method
                    ref_: reference values obtained from bagging or random forest method
                Returns: numeric array of predictions, dictionary of form {class:[output values]
                """
        # dict for input classification
        Y_pred = np.zeros((self.B, input_.shape[0]))
        predictors = list(input_.columns)

        for b in tqdm(range(self.B)):
            for i in model[b]:
                mask = input_.copy()
                for j in predictors:
                    mask[j] = (input_[j] >= model[b][i][j][0]) & \
                              (input_[j] <= model[b][i][j][1])
                row_ind = np.argwhere(np.array(mask.all(axis=1)) == True)
                Y_pred[b, row_ind] = ref_[b][i]

        Y_pred_ = np.ones(Y_pred.shape[1])
        if self.criterion == 'RSS':
            # average value of B bootstrap predictions for regression problems
            Y_pred_ = np.average(Y_pred, axis=0)
        else:
            # most common vote for B bootstrap predictions for classification problems
            for i in range(Y_pred.shape[1]):
                Y_pred_[i] = Y_pred[np.argmax(np.unique(Y_pred[:, i], return_counts=True)[1]), i]
        return Y_pred_

    def bagging_predict(self, input_):
        return self.predict_bag_rf(input_=input_, model=self.bagging_model, ref_=self.bagging_ref)

    def bootstrapping(self, B):
        """
        Function generates B bootstrap training data sets of size n.
        Args:
            B: number of bootstrap training sets

        Returns: numeric array of training observation indices of shape (B×n)

        """
        n = self.training_inputs.shape[0]
        bootstrap_ind = np.zeros((B, n), dtype=int)
        for b in range(B):
            bootstrap_ind[b, :] = np.random.randint(0, n, n)
        return bootstrap_ind

    # TODO add error estimation (Out-Of-Bag / OOB)
    # Random Forests
    def random_forest(self, B, m=None):
        """
        Function generates series of decision trees using bootstrapped training data.
        Args:
            B: number of trees generated (int)
            m: number of predictors in individual split (int)

        Returns: list of decision trees
        """
        if m is None:
            m = math.ceil(np.sqrt(len(self.training_inputs.columns)))

        self.B = B
        self.random_f_model = {}
        self.random_f_ref = {}
        bootstrap_ind = self.bootstrapping(B=B)

        for i in tqdm(range(B)):
            # print(i)
            self.random_f_model[i], self.random_f_ref[i], _, _ = self.generate_tree(
                self.training_inputs.iloc[bootstrap_ind[i, :], :],
                self.training_outputs[bootstrap_ind[i, :]], m=m)

    def random_forest_predict(self, input_):
        return self.predict_bag_rf(input_=input_, model=self.random_f_model, ref_=self.random_f_ref)


def k_fold_CV_for_decision_tree_pruning(inputs_, outputs_, alpha, criterion,
                                        sizes, k=10,
                                        find_alpha_opt=False, r_size=10,
                                        ):
    # data split
    input_f, output_f = ms.k_fold_split(np.array(inputs_), np.array(outputs_), k=k)
    error_rates = []
    for alpha_ind, k_ in enumerate(tqdm(alpha, desc=f'Main loop')):
        errors = []
        for i in tqdm(range(k), desc=f'loop for α = {k_:.4f}: ', leave=False):
            pred_fold_ = pd.DataFrame(input_f[i])
            ref_fold_ = output_f[i]
            ind_list = list(np.arange(k))
            ind_list.remove(i)
            input_folds_ = np.array(input_f, dtype=object)[ind_list]
            input_folds_ = np.concatenate(input_folds_.squeeze(), axis=0)
            output_folds_ = np.array(output_f, dtype=object)[ind_list]
            output_folds_ = np.concatenate(output_folds_.squeeze(), axis=0)
            input_f_df = pd.DataFrame(input_folds_)
            output_f_s = pd.Series(output_folds_)
            # form the decision tree for k-th fold
            my_tree = DecisionTree(input_f_df, output_f_s, s_div=30, max_leaf_size=r_size,
                                   criterion=criterion)
            # shrink the tree for current α
            my_tree.shrink_tree(size_goal=sizes[alpha_ind])
            y_pred, _ = my_tree.predict(pred_fold_)
            errors.append(ms.RSS(y_pred, ref_fold_))
        error_rates.append(np.average(errors))
    if find_alpha_opt:
        alpha_opt = alpha[np.argmin(np.array(error_rates))]
        return error_rates, alpha_opt
    return error_rates


h_df = pd.read_csv('C:/Users/timvr/Documents/Doktorat/Introduction to statistical Learning/Gradivo/Heart.csv')
h_df.drop(labels='Unnamed: 0', axis=1, inplace=True)
features = ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR',
            'ExAng', 'Oldpeak', 'Slope', 'Ca']
HD_dummy = np.zeros(h_df.shape[0])
HD_dummy[h_df.AHD == 'Yes'] = 1
HD_dummy[h_df.AHD == 'No'] = 0
my_tree = DecisionTree(h_df[features][1::2], HD_dummy[1::2], s_div=20, max_leaf_size=10, criterion='Gini')

# def boosting(x_in, y_in, d, B):

#    tree = DecisionTree()
# y_cl_pred, y_cl = my_tree.predict(h_df[features][::2])
# my_tree.bagging(B=30)
# y_pred_bagging = my_tree.bagging_predict(h_df[features][::2])
# ms.confusion_matrix(y_pred_bagging, HD_dummy[::2], print_=True)
# print(f'Error rate: {ms.classification_error_rate(y_pred_bagging, HD_dummy[::2]):.3f}')
# my_tree.random_forest(B=100)
# y_pred_rf = my_tree.random_forest_predict(h_df[features][::2])
# ms.confusion_matrix(y_pred_rf, HD_dummy[::2], print_=True)
# print(f'Error rate: {ms.classification_error_rate(y_pred_rf, HD_dummy[::2]):.3f}')
# alphas = [0., 0.5052631578947369, 0.5894736842105263, 0.7578947368421053]
# sizes_ = [58, 56, 40, 28]
# r_size_ = 10

# errors_ = k_fold_CV_for_decision_tree_pruning(h_df[features], HD_dummy, k=5, alpha=alphas, sizes=sizes_,
#                                               r_size=r_size_, criterion='gini')


