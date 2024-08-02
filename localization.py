import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment

# conditional independence test (KCI Test)
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test as hsic
from conditional_independence import partial_correlation_test as partial_corr
from conditional_independence import partial_correlation_suffstat as partial_corr_stat


class IntervLocalizer:
    def __init__(self, w=None, observation=None, rec_noise=None, label=None, a_full=None):
        self.a_full = a_full    # True adjacency matrix ([non-leaf, leaf])
        if a_full is not None:
            self.a = a_full[:a_full.shape[1]]  # Non-leaf adj matrix
            self.b = a_full[a_full.shape[1]:]  # Leaf adj matrix
        else:
            self.a, self.b = None, None

        if w is None:
            self.w = np.vstack((np.eye(a_full.shape[1]), self.b)) @ np.linalg.inv(np.eye(a_full.shape[1]) - self.a)
        else:
            self.w = w
        self.w_me = None
        self.p = self.w.shape[0]
        self.p_t = None  # number of variables in T

        self.order = None
        self.rev_order = None
        self.unique = None

        self.observation = observation
        self.noise = rec_noise
        self.label=label
        self.unique_label = np.unique(label)

    def normalize(self, alpha=5e-2, prune=False):
        '''
        Prune the mixing matrix.
        '''
        # pre-processing
        w_me = self.w.copy()
        p = w_me.shape[0]
        m = w_me.shape[1]
        # w_me = w_me / np.maximum(np.linalg.norm(w_me, axis=0), 1e-6)  # column noralization
        w_me = w_me / np.sign(w_me[np.argmax(abs(w_me), axis=0), range(m)])  # flip sign
        w_me[np.abs(w_me) <= alpha] = 0 # initlal  prune
        self.w_me = w_me
        self.p_t = self.w_me.shape[1]  # number of variables in T

        # Pruning
        if prune:
            idx_abs = np.argsort(np.abs(w_me), axis=None)
            for i in idx_abs:
                self.w_me[np.unravel_index(i, w_me.shape)] = 0
                _, _, col_order = self.interv_nolatent(ci_test=False, permute_column=True)
                # print(col_order)
                if len(col_order) == self.p_t:
                    break

    def find_possible_parent(self, target, row_index):
        '''
        Find the possible parent set for each target variable.
        '''
        pp = []
        tgt_row = self.w_me[target]
        var_set = [ _ for _ in range(self.p) if _ not in row_index ]
        for var in var_set:
            if tgt_row[self.w_me[var] != 0].all():
                pp.append(var)
        return pp
    
    def interv_nolatent(self, ci_test='hsic', permute_column=False):
        '''
        Return the intervention target using the (pruned) mixing matrix under causal sufficiency.
        Input:
            ci_tests: Method for CI test
            permute_column: If we want to return the column order.
        Output:
            int_set: Set of intervention targets.
            counter: Total number of CI tests (excluding the tests for finding the indicator set). 
            col_order: Order of the columns.
        '''
        # Input: mixing matrix w
        nnz_row = np.count_nonzero(self.w_me, axis=1)
        row_index = np.arange(self.p)
        col_index = np.arange(self.p_t)
        int_set = []
        col_order = []

        # remove all-zero rows
        temp_index = np.where(nnz_row == 0)[0]
        row_index = np.delete(row_index, temp_index)
        nnz_row = np.delete(nnz_row, temp_index)
        counter = 0

        for _ in range(self.p_t):
            w_temp = self.w_me[row_index][:, col_index]
            temp_index = np.where(np.count_nonzero(w_temp, axis=1) == 1)[0]
            if len(temp_index) == 0:
                break
            row = temp_index[nnz_row[temp_index].argmin()]
            n_0 = nnz_row[row]
            col = np.where(w_temp[row] != 0)[0][0]

            z_i = temp_index[w_temp[temp_index, col] != 0]  # Z_I
            z_j = z_i[nnz_row[z_i] == n_0]

            p_j = len(z_j)
            if p_j == 1:
                int_set.append(row_index[z_j[0]])
            else:
                ##################################################
                # Find the variable that satisfies condition (III)
                # independent means larger p-value
                temp_noise = self.noise[:, col_index[col]]
                temp_var = self.observation[:, row_index[z_j]]
                parent_set = self.observation[:, self.find_possible_parent(row_index[z_j[0]], row_index)]
                temp_data = np.hstack((temp_var, parent_set, temp_noise[:, np.newaxis]))
                cond_set = list(range(p_j, temp_data.shape[1] - 1))
                citest_optim = -1
                var_optim = row_index[z_j[0]]
                for var_i in range(p_j):
                    citest_temp = 0.
                    cond_set_tmp = cond_set + [var_i]
                    for var_j in range(p_j):
                        if var_j != var_i:
                            counter += 1 # number of CI tests
                            for l in self.unique_label:
                                if ci_test == 'hsic':
                                    citest_temp += hsic(temp_data[self.label == l], -1, var_j, cond_set_tmp)['p_value']
                                elif ci_test == 'partial_corr':
                                    corr_dict = partial_corr_stat(temp_data[self.label == l])
                                    citest_temp += partial_corr(corr_dict, -1, var_j, cond_set_tmp)['p_value']
                        # print(hsic_temp)
                    if citest_temp >= citest_optim:
                        citest_optim = citest_temp
                        var_optim = row_index[z_j[var_i]]
                int_set.append(var_optim)

            col_order.append(col_index[col])

            row_index = np.delete(row_index, z_i)
            col_index = np.delete(col_index, col)
            nnz_row = np.delete(nnz_row, z_i) # Also update nnz

        return int_set, counter, col_order if permute_column else None

    def interv_latent(self, ci_test='hsic', permute_column=False, threshold=0.05):
        '''
        Return the intervention target using the (pruned) mixing matrix under the presense of latent confounding.
        Input:
            ci_tests: Method for CI test
            permute_column: If we want to return the column order.
            threshold: Threshold used for Condition (III-L).
        Output:
            int_set: Set of intervention targets.
            counter: Total number of CI tests (excluding the tests for finding the indicator set). 
            col_order: Order of the columns.
        '''
        nnz_row = np.count_nonzero(self.w_me, axis=1)
        nnz_col = np.count_nonzero(self.w_me, axis=0)
        row_index = np.arange(self.p)
        col_index = np.arange(self.p_t)
        int_set = []
        latent_noise = []
        col_order = []

        # remove rows with all zero
        temp_index = np.where(nnz_row == 0)[0]
        row_index = np.delete(row_index, temp_index)
        nnz_row = np.delete(nnz_row, temp_index)

        counter = 0

        while len(col_index) > 0:
            w_temp = self.w_me[row_index][:, col_index]
            count = np.count_nonzero(w_temp, axis=1)
            if (count < 1).all():
                break
            temp_row = np.where(count == count.min())[0]  # row with the fewest number of non-zero entry, index
            # print(temp_row)
            row = temp_row[nnz_row[temp_row].argmin()]  # selected row, index
            n_0 = nnz_row[row]

            N_i = np.where(w_temp[row] != 0)[0]  # index
            N_j = N_i[nnz_col[N_i] == nnz_col[N_i].min()]
            latent_noise += [ col_index[_] for _ in N_i if _ not in N_j]

            # print(w_temp[temp_row])
            z_i = temp_row[w_temp[temp_row][:, N_i].all(axis=1)]  # rows with the same support as "row", index
            z_j = z_i[nnz_row[z_i] == n_0]
            # print(z_i)

            # Line 7
            col = N_j[0]
            row_select = np.where(self.w_me[:, col_index[col]] != 0)[0]
            col_select = np.where(self.w_me[row_index[row]] != 0)[0]
            col_select = [ ii for ii in col_select if ii not in latent_noise]
            w_check = self.w_me[row_select][:, col_select]
            if not w_check.all(axis=None): # The matrix is not all-zero
                latent_noise += [ col_index[_] for _ in N_j]
            else:
                p_j = len(z_j)
                if p_j == 1:
                    int_set.append(row_index[z_j[0]])
                else:
                    ##################################################
                    # Find the variable that satisfies condition (III)
                    temp_noise = self.noise[:, latent_noise + list(col_index[N_j])]
                    temp_var = self.observation[:, row_index[z_j]]
                    parent_set = self.observation[:, self.find_possible_parent(row_index[z_j[0]], row_index)]
                    if len(latent_noise) + len(N_j) == 1:
                        temp_data = np.hstack((temp_var, parent_set, temp_noise))
                    else:
                        temp_data = np.hstack((temp_var, parent_set, temp_noise))
                    
                    cond_set = list(range(p_j, temp_data.shape[1] - 1))
                    citest_optim = -1
                    var_idx = list(range(p_j))
                    for var_i in range(p_j):
                        citest_temp = 0.
                        cond_set_tmp = cond_set + [var_i]
                        for var_j in range(p_j):
                            if var_j != var_i:
                                counter += 1
                                for l in self.unique_label:
                                    if ci_test == 'hsic':
                                        citest_temp += hsic(temp_data[self.label == l], -1, var_j, cond_set_tmp)['p_value']
                                    elif ci_test == 'partial_corr':
                                        corr_dict = partial_corr_stat(temp_data[self.label == l])
                                        citest_temp += partial_corr(corr_dict, -1, var_j, cond_set_tmp)['p_value']
                        citest_temp /= ((p_j-1) * len(self.unique_label))
                        # print(citest_temp)
                        if citest_temp <= threshold:
                            var_idx.remove(var_i)

                        # if citest_temp >= citest_optim:
                        #     citest_optim = citest_temp
                        #     var_optim = row_index[z_j[var_i]]
                    for ii in row_index[z_j[var_idx]]:
                        int_set.append(ii)

            if len(z_i) == 1:
                col_order.append(col_index[col])
            else:
                col_order += [ col_index[_] for _ in N_i]

            row_index = np.delete(row_index, z_i)
            col_index = np.delete(col_index, N_i)
            nnz_row = np.delete(nnz_row, z_i)
            nnz_col = np.delete(nnz_col, N_i)

        return int_set, counter, col_order if permute_column else None        


if __name__ == '__main__':
    W = np.array([[1, 0, 0.],
                  [1, 1, 0.],
                  [0, 1, 1.]])

    model = IntervLocalizer(w=W)
    model.normalize(prune=False)
    print(model.w_me)
    aog, bb = model.interv_latent(permute_column=True)
    print(aog, bb)