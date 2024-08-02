from .munkres import Munkres
import numpy as np
import scipy as sp
import sklearn

from scipy.stats import spearmanr, ks_2samp, wasserstein_distance
from scipy.optimize import linear_sum_assignment


def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    print("Calculating correlation...")

    x = np.array(x).copy()
    y = np.array(y).copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method == 'Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim, dim:]
    elif method == 'Spearman':
        corr, pvalue = sp.stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i, :] = x[indexes[i][1], :]

    # Re-calculate correlation --------------------------------
    if method == 'Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim, dim:]
    elif method == 'Spearman':
        corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort


def SolveHungarian(x, y, correlation='Pearson'):
    """
    compute maximum correlations between true indep components and estimated components 
    """
    Ncomp = x.shape[1]
    if correlation == 'Pearson':
        CorMat = (np.abs(np.corrcoef(x.T, y.T)))[:Ncomp, Ncomp:]
    elif correlation == 'Spearman':
        rho, _ = np.abs(spearmanr(x, y))
        CorMat = rho[:Ncomp, Ncomp:]
    ii = linear_sum_assignment(CorMat, maximize=True)

    return CorMat[ii].mean(), CorMat, ii


def get_correlation_metrics(xseval, feat_val, targets):
    """
    get the metrics for correlation when optimally matching
    
    also includes top k metrics:
        if we look at the top k correlated variables,
        where (for now) k := number of environments (interventions)

    compute dict with
        * corr avg
        * accuracy between top k intervention targets
          and top k correlated variables when seen as binary classification
        * recall
        * correlation if we only take the top k correlated variables

    """
    metrics = {}
    num_vars = xseval.shape[1]
    rec_vars = feat_val.shape[1]
    corr_avg, mat, ii = SolveHungarian(xseval, feat_val)

    # correlation for each optimally matched var
    for i in range(rec_vars):
        metrics["corr_" + str(i)] = mat[i, ii[1][i]]

    unique_targets = np.unique(targets)
    k = len(unique_targets)
    top_k_vars = np.argsort(mat[ii])[-k:]
    top_k_corr_avg = np.sort(mat[ii])[-k:].mean()

    real_target_mask = np.zeros(num_vars)
    real_target_mask[unique_targets] += 1
    top_k_target_mask = np.zeros(num_vars)
    top_k_target_mask[top_k_vars] += 1

    top_k_acc = (real_target_mask == top_k_target_mask).mean()
    top_k_recall = sklearn.metrics.recall_score(real_target_mask,
                                                top_k_target_mask)
    return dict(**metrics,
                corr_avg=corr_avg,
                top_k_acc=top_k_acc,
                top_k_recall=top_k_recall,
                top_k_corr_avg=top_k_corr_avg)


def find_source_change_metric(label, feat_val, num_env, num_vars):
    ws = []
    for var in range(num_vars):
        ws_ = []
        for e in range(num_env-1):
            x = feat_val[label == 0, var]
            y = feat_val[label == e + 1, var]
            # print("e={}, i={}, pval={:.6f}".format(j + 1, var, ks_2samp(*x).pvalue))
            ws_.append(wasserstein_distance(x, y))
        ws.append(sum(ws_)/len(ws_))
    ws = np.array(ws)
    return ws




def get_env_change_metrics(targets, label, xseval, feat_val):
    """
    compute metrics for identifying the changing noises across environments
    with the recovered sources
    
    based on two methods:
        1) minimal pvalue of KS 2 sample test
        2) maximal wasserstein distance
    for now, difference is taken to observational environment, i.e. environment 0 
    """
    num_env = len(targets)
    num_vars = xseval.shape[1]

    # just need matching ii from correct source -> recovered index
    _, _, ii = SolveHungarian(xseval, feat_val)
    ps = []
    for e in range(num_env):
        ps_ = []
        for var in range(num_vars):
            x = feat_val[label == 0, ii[1][var]]
            y = feat_val[label == e + 1, ii[1][var]]
            # print("e={}, i={}, pval={:.6f}".format(j + 1, var, ks_2samp(*x).pvalue))
            ps_.append([ks_2samp(x, y).pvalue, wasserstein_distance(x, y)])
        ps.append(ps_)
    ps = np.array(ps)
    print(ps)
    #print(targets)
    #print(ps[..., 0].argmin(1)) #[2 4 1 3 0]
    ks_test_acc = (targets == ps[..., 0].argmin(1)).mean()
    wasserstein_acc = (targets == ps[..., 1].argmax(1)).mean()

    return dict(ks_test_acc=ks_test_acc, wasserstein_acc=wasserstein_acc)
