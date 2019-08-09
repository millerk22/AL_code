import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import newton, root_scalar
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csgraph
import time
from heapq import *
from sklearn.datasets import make_moons
import copy



def generate_data_graphs(Ns, means, Covs, k_nn=5):
    '''
    inputs   :
      Ns    : class sizes, an array of length K
      means : class means, list of mean vectors in R^d
      Covs  : class covariances, list of Cov matrices of size d x d
    returns  : adjacency matrix
    '''
    N = sum(Ns)
    d = len(means[0])
    X = np.zeros((N, d))
    offset = 0
    for i in range(len(Ns)):
        Ni = Ns[i]
        X[offset:offset+Ni,:] = np.random.multivariate_normal(means[i], Covs[i], Ni)
        offset += Ni


    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]

    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    w_sp = np.divide(Dknn, Dmean)
    w_sp = np.exp(-(w_sp * w_sp))
    W = sps.csr_matrix((w_sp.flatten() , (I, J)), shape=(N,N))
    W = 0.5*(W+W.T)

    return X, W

def sqdist(X, Y):
    # Michael Luo code  - X is (d x m) np array, Y is (d x n) np array
    # returns D_ij = |x_i - y_j|^2 matrix
    # better than KD tree for larger dimensions?
    m = X.shape[1]
    n = Y.shape[1]
    Yt = Y.T
    XX = np.sum(X*X, axis=0)
    YY = np.sum(Yt*Yt, axis=1).reshape(n,1)
    return np.tile(XX, (n,1)) + np.tile(YY, (1,m)) - 2*Yt.dot(X)



# Main setup, defining the node set, fidelity nodes, ground truth u and observations y, and corresponding
# B and H matrices

def other_setup(Ns=[100,100,100],num_in_each=5):
    N = sum(Ns)

    # begin with ground truth, u
    fid = np.zeros(N, dtype='int')
    idx = 0
    for l in range(len(Ns)):
        fid[idx:idx+num_in_each] = 1
        idx += Ns[l]

    B = sp.sparse.diags(fid, format='lil')
    labeled = np.nonzero(fid)[0]


    ground_truth = {i:1. if i <= Ns[0] else -1. for i in range(N)}

    return B, labeled, ground_truth

def compute_laplacian(W):
    D = sp.sparse.diags(np.sum(W, axis=0).getA().flatten())
    L = D - W
    return L, D.diagonal()


def get_eig_Lnorm(W):
    L_sym = csgraph.laplacian(W, normed=True)
    [w, v] = sp.linalg.eigh(L_sym.toarray())

    return w, v


########################## Single Updates #######################

def calc_next_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[0,k]
    ip = np.dot(ck[0,lab], y[lab,np.newaxis])[0,0]
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k

def calc_next_C_and_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[0,k]
    # calculate m_k
    ip = np.dot(ck[0,lab], y[lab,np.newaxis])[0,0]
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck


    # calculate C_k -- the posterior of adding k, y_k
    C_k = C - (1./(gamma2 + ckk))*np.outer(ck,ck)

    return m_k, C_k


def get_probs(m):
    m_probs = m.flatten() # simple fix to get probabilities that respect the decision bdry
    m_probs[np.where(m_probs >0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs <0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs


############### EEM Risk #########
def calc_risk(k, m, C, y, lab, unlab, m_probs, gamma2):
    m_at_k = m_probs[0,k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = (1. - m_at_k)*np.sum([min(m_k_p1[0,i], 1.- m_k_p1[0,i]) for i in unlab])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += m_at_k*np.sum([min(m_k_m1[0,i], 1.- m_k_m1[0,i]) for i in unlab])
    return risk




############## Active Learning Functions for Gaussian Clusters #######

def calc_orig(v, w, B, Ns, labeled, tau, alpha, gamma2):
    N = sum(Ns)
    sup1 = labeled[labeled < Ns[0]]
    sup2 = labeled[labeled >= Ns[0]]
    y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
    y[sup1] = 1.
    y[sup2] = -1.

    d = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    # prior_inv : C_{tau,eps}^{-1}, where
    # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
    prior_inv = v.dot(sp.sparse.diags([1./thing for thing in d], format='lil').dot(v.T))
    # B/gamma^2
    B_over_gamma2 = B / (gamma2)
    # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
    post_inv  = prior_inv + B_over_gamma2
    C = post_inv.I
    m = (1./gamma2)*C.dot(y).flatten()

    return m, C, y



def run_next(m, C, y, lab, unlab, ground_truth, gamma2):
    risks = []
    m_probs = get_probs(m)
    for j in unlab:
        risk_j = calc_risk(j, m, C, y, lab, unlab, m_probs.copy(), gamma2)
        heappush(risks, (risk_j, j))

    k_next_risk, k_next = heappop(risks)
    # Ask "the oracle" for k_next's value, known from ground truth in Ns
    y_k_next = ground_truth[k_next]

    m_next, C_next = calc_next_C_and_m(m, C, y, lab, k_next, y_k_next, gamma2)
    y_next = y.copy()
    y_next[k_next] = y_k_next
    lab = np.array(list(lab)+ [k_next])
    unlab.remove(k_next)

    return k_next, m_next, C_next, y_next, lab, unlab




def calc_stats(m, labeled, Ns=[100,100], _print=False):
    stats = {}
    N = sum(Ns)
    m1 = np.where(m >= 0)[1]
    m2 = np.where(m < 0)[1]
    sup1 = labeled[labeled < Ns[0]]
    sup2 = labeled[labeled >= Ns[0]]
    corr1 = list(set(m1).intersection(set(range(0,Ns[0]))))
    incorr1 = list(set(m2).intersection(set(range(0,Ns[0]))))
    corr2 = list(set(m2).intersection(set(range(Ns[0],N))))
    incorr2 = list(set(m1).intersection(set(range(Ns[0],N))))

    stats['corr1'] = corr1
    stats['corr2'] = corr2
    stats['sup1'] = sup1
    stats['sup2'] = sup2
    stats['incorr1'] = incorr1
    stats['incorr2'] = incorr2

    error = ((len(incorr1) + len(incorr2))/N )
    if _print:
        print('Error = %f' % error )

    return error, stats




def plot_iter(stats, X, k_next=-1):
    corr1 = stats['corr1']
    corr2 = stats['corr2']
    sup1 = stats['sup1']
    sup2 = stats['sup2']
    incorr1 = stats['incorr1']
    incorr2 = stats['incorr2']

    if k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new point to be included
    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='b', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='r', alpha=1.0)
    plt.axis('equal')
    plt.title('Dataset with Label for %d added' % k_next)
    plt.show()
    return








    ########################### 2 Moons Functions

def make_sim_graph(X, k_nn=5):
    N = X.shape[0]
    # Make weighted similarity graph, in W
    D = sqdist(X.T,X.T)
    ind_knn = np.argsort(D, axis=1)[:,1:k_nn+1]
    Dknn = D[(np.arange(N).reshape(N,1),ind_knn)]

    I = np.tile(np.arange(N).reshape(N,1), (1,k_nn)).flatten()
    J = ind_knn.flatten()
    Dmean = (np.sum(Dknn, axis=1)/k_nn).reshape(N,1)
    w_sp = np.divide(Dknn, Dmean)
    w_sp = np.exp(-(w_sp * w_sp))
    W = sps.csr_matrix((w_sp.flatten() , (I, J)), shape=(N,N))
    W = 0.5*(W+W.T)

    return W

def calc_orig2(v, w, B, fid, tau, alpha, gamma2):
    N = v.shape[0]
    y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
    y[fid[1]] = 1.
    y[fid[-1]] = -1.

    d = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    # prior_inv : C_{tau,eps}^{-1}, where
    # C_{tau, eps}^{-1} = tau^{-2alpha}(L + tau^2 I)^alpha
    prior_inv = v.dot(sp.sparse.diags([1./thing for thing in d], format='lil').dot(v.T))
    # B/gamma^2
    B_over_gamma2 = B / (gamma2)
    # post_inv  : (B/gamma^2 + C_{tau,\eps}^{-1})^{-1}
    post_inv  = prior_inv + B_over_gamma2
    C = post_inv.I
    m = (1./gamma2)*C.dot(y).flatten()

    return m, C, y



def run_next2(m, C, y, lab, unlab, fid, ground_truth, gamma2):
    tic = time.clock()
    risks = []
    for j in unlab:
        risk_j = calc_risk(j, m, C, y, lab, unlab, gamma2)
        heappush(risks, (risk_j, j))

    k_next_risk, k_next = heappop(risks)
    toc = time.clock()
    print('Time for EEM = %f' % (toc - tic))

    # Ask "the oracle" for k_next's value, known from ground truth in Ns
    y_k_next = ground_truth[k_next]
    fid[y_k_next].append(k_next)

    m_next, C_next = calc_next_C_and_m(m, C, y, lab, k_next, y_k_next, gamma2)
    y[k_next] = y_k_next
    lab = np.array(list(lab)+ [k_next])
    unlab.remove(k_next)


    return k_next, m_next, C_next, y, lab, unlab, fid




def calc_stats2(m, fid, gt_flipped, _print=False):
    stats = {}
    N = m.shape[1]
    m1 = np.where(m >= 0)[1]
    m2 = np.where(m < 0)[1]
    sup1 = fid[1]
    sup2 = fid[-1]
    corr1 = list(set(m1).intersection(set(gt_flipped[1])))
    incorr1 = list(set(m2).intersection(set(gt_flipped[1])))
    corr2 = list(set(m2).intersection(set(gt_flipped[-1])))
    incorr2 = list(set(m1).intersection(set(gt_flipped[-1])))

    stats['corr1'] = corr1
    stats['corr2'] = corr2
    stats['sup1'] = sup1
    stats['sup2'] = sup2
    stats['incorr1'] = incorr1
    stats['incorr2'] = incorr2

    error = ((len(incorr1) + len(incorr2))/N )
    if _print:
        print('Error = %f' % error )

    return error, stats

def run_test_AL(X, v, w, labeled, unlabeled, fid, ground_truth, gt_flipped, tau, alpha, gamma2, iters=5, verbose=False):
    N = len(ground_truth)
    B_diag = np.zeros(N)
    B_diag[labeled] = 1.
    B = sp.sparse.diags(B_diag, format='lil')

    m, C, y = calc_orig2(v, w, B, fid, tau, alpha, gamma2)

    ERRS = []
    error, stats_obj = calc_stats2(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        plot_iter(stats_obj, X, k_next=-1)

    M = {}
    M[-1] = m

    # AL choices
    for l in range(iters):
        k, m, C, y, labeled, unlabeled, fid = run_next2(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2)
        error, stats_obj = calc_stats2(m, fid, gt_flipped)
        ERRS.append((k,error))
        M[l] = m
        if verbose:
            print('Iter = %d' % (l + 1))
            plot_iter(stats_obj, X, k_next=k)
    return ERRS, M




def run_test_rand(X, v, w, labeled, unlabeled, fid, ground_truth, gt_flipped, tau, alpha, gamma2, iters=10, verbose=False):
    N = len(ground_truth)
    B_diag = np.zeros(N)
    B_diag[labeled] = 1.
    B = sp.sparse.diags(B_diag, format='lil')

    m, C, y = calc_orig2(v, w, B, fid, tau, alpha, gamma2)

    ERRS_rand = []
    error, stats_obj = calc_stats2(m, fid, gt_flipped)
    ERRS_rand.append((-1,error))
    if verbose:
        print('Iter = 0')
        plot_iter(stats_obj, X, k_next=-1)

    M_rand = {}
    M_rand[-1] = m

    # Rand choices
    for l in range(iters):

        k_next = unlabeled[np.random.choice(len(unlabeled),1)[0]]
        y_k_next = ground_truth[k_next]
        y[k_next] = y_k_next

        m, C = calc_next_C_and_m(m, C, y, labeled, k_next, y_k_next, gamma2)
        labeled = np.array(list(labeled)+ [k_next])
        unlabeled.remove(k_next)

        M_rand[l] = m
        error, stats_obj = calc_stats2(m, fid, gt_flipped)
        ERRS_rand.append((k_next, error))
        if verbose:
            print('Iter = %d' % (l + 1))
            plot_iter(stats_obj, X, k_next=k_next)
    return ERRS_rand, M_rand






    ################# KL Functions for 2 Moons #####
    def run_next2_KL(m, C, y, lab, unlab, fid, ground_truth, gamma2, W):
        m_probs = m.copy() # simple fix to get probabilities that respect the decision bdry
        m_probs[m_probs >0] /= 2.*np.max(m_probs)
        m_probs[m_probs <0] /= -2.*np.min(m_probs)
        m_probs += 0.5
        calc_class1_ind = np.where(m_probs >= 0.5)[0]
        calc_class2_ind = np.where(m_probs < 0.5)[0]
        tic = time.clock()
        nz_entries
        toc = time.clock()
        print('Time for KL = %f' % (toc - tic))


        # Ask "the oracle" for k_next's value, known from ground truth in Ns
        y_k_next = ground_truth[k_next]
        fid[y_k_next].append(k_next)

        m_next, C_next = calc_next_C_and_m(m, C, y, labeled, k_next, y_k_next, gamma2)
        y[k_next] = y_k_next
        lab = np.array(list(lab)+ [k_next])
        unlab.remove(k_next)


        return k_next, m_next, C_next, y, lab, unlab, fid


    def run_test_AL2_KL(X, v, w, W, labeled, unlabeled, fid, ground_truth, gt_flipped, tau, alpha, gamma2, iters=5, verbose=False):
        N = len(ground_truth)
        B_diag = np.zeros(N)
        B_diag[labeled] = 1.
        B = sp.sparse.diags(B_diag, format='lil')

        m, C, y = calc_orig2(v, w, B, fid, tau, alpha, gamma2)

        ERRS = []
        error, stats_obj = calc_stats2(m, fid, gt_flipped)
        ERRS.append((-1,error))
        if verbose:
            print('Iter = 0')
            plot_iter(stats_obj, X, k_next=-1)

        M = {}
        M[-1] = m

        # AL choices
        for l in range(iters):
            k, m, C, y, labeled, unlabeled, fid = run_next2_KL(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2, W)
            error, stats_obj = calc_stats2(m, fid, gt_flipped)
            ERRS.append((k,error))
            M[l] = m
            if verbose:
                print('Iter = %d' % (l + 1))
                plot_iter(stats_obj, X, k_next=k)
        return ERRS, M
