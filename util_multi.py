import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import newton, root_scalar
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csgraph
import time
from heapq import *
from sklearn.datasets import make_moons
import copy
from itertools import permutations
from sklearn.datasets import make_moons


################# Graph generation and other Calculations

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

# Making a similarity graph
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


# Gaussian clusters data
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
    W = make_sim_graph(X, k_nn)
    return X, W


def get_eig_Lnorm(W, return_L=False):
    L_sym = csgraph.laplacian(W, normed=True)
    [w, v] = sp.linalg.eigh(L_sym.toarray())
    if return_L:
        return w, v, L_sym
    return w, v



########################## Single Updates #######################

def calc_next_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck
    return m_k

def calc_next_C_and_m(m, C, y, lab, k, y_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ip = np.dot(ck[lab], y[lab])
    val = ((gamma2)*y_k -ip )/(gamma2*(gamma2 + ckk))
    m_k = m + val*ck

    # calculate C_k -- the posterior of adding k, y_k
    C_k = C - (1./(gamma2 + ckk))*np.outer(ck,ck)
    return m_k, C_k

def calc_next_C_and_m_multi(m, C, y, lab, k, class_ind_k, gamma2):
    ck = C[k,:]
    ckk = ck[k]
    ec = np.zeros(y.shape[1])
    ec[class_ind_k] = 1.
    ip = np.dot(ck[lab], y[lab])
    print(ip.shape)
    print(ec.shape)
    outer_term = (ec - (ip.T/gamma2))/(gamma2 + ckk)
    print(outer_term)
    m_k = m + np.outer(ck,outer_term)

    # calculate C_k -- the posterior of adding k, y_k
    C_k = C - (1./(gamma2 + ckk))*np.outer(ck,ck)
    return m_k, C_k

def calc_next_m_batch(m, C, y, lab, k_to_add, y_ks, gamma2):
    C_b = C[:, k_to_add]
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[k_to_add] = y_ks
    m_next = m + C_b.dot(y_ks)/gamma2
    C_bb_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + C_b[k_to_add,:])
    m_next -= (1./gamma2)*C_b.dot(C_bb_inv.dot(C_b[lab_new,:].T.dot(y_next[lab_new])))
    return m_next

def calc_next_C_and_m_batch(m, C, y, lab, k_to_add, y_ks, gamma2):
    Cb = C[:,k_to_add]
    mat_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + Cb[k_to_add,:])
    C -= Cb.dot(mat_inv.dot(Cb.T))

    # Update m now
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[k_to_add] = y_ks
    m_batch = (1./gamma2)*C[:,lab_new].dot(y_next[lab_new])

    return m_batch, C

def calc_next_m_batch_multi(m, C, y, lab, k_to_add, class_ind_ks, gamma2):
    C_b = C[:, k_to_add]
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[np.ix_(k_to_add, class_ind_ks)] = 1.
    m_next = m + C_b.dot(y_next[k_to_add,:])/gamma2
    C_bb_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + C_b[k_to_add,:])
    m_next -= (1./gamma2)*C_b.dot(C_bb_inv.dot(C_b[lab_new,:].T.dot(y_next[lab_new,:])))
    return m_next, y_next, lab_new

def calc_next_C_and_m_batch_multi(m, C, y, lab, k_to_add, class_ind_ks, gamma2):
    Cb = C[:,k_to_add]
    mat_inv = sla.inv(gamma2*np.eye(len(k_to_add)) + Cb[k_to_add,:])
    C -= Cb.dot(mat_inv.dot(Cb.T))

    # Update m now
    lab_new = lab[:]
    lab_new.extend(k_to_add)
    y_next = y.copy()
    y_next[np.ix_(k_to_add,class_ind_ks)] = 1.
    m_batch = (1./gamma2)*C[:,lab_new].dot(y_next[lab_new,:])

    return m_batch, C, y_next, lab_new

# Transform the vector m into probabilities, while still respecting the threshold value currently at 0
def get_probs(m):
    m_probs = m.copy()
    #m_probs = m.flatten() # simple fix to get probabilities that respect the decision bdry
    m_probs[np.where(m_probs >0)] /= 2.*np.max(m_probs)
    m_probs[np.where(m_probs <0)] /= -2.*np.min(m_probs)
    m_probs += 0.5
    return m_probs

# Transform the matrix m into probabilities, while still respecting the threshold
def get_probs_multi(m):
    m_probs = m.copy()
    #m_probs = m.flatten() # simple fix to get probabilities that respect the decision bdry
    m_probs -= 0.5
    for j in range(m.shape[1]): # for each class vector, normalize to be probability in 0,1, respecting 0.5 threshold
        m_probs[np.where(m_probs[:,j] >0),j] /= 2.*np.max(m_probs[:,j])
        m_probs[np.where(m_probs[:,j] <0),j] /= -2.*np.min(m_probs[:,j])
    m_probs += 0.5



    return m_probs


############### EEM Risk #########
def calc_risk(k, m, C, y, lab, unlab, m_probs, gamma2):
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = (1. - m_at_k)*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in unlab])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += m_at_k*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in unlab])
    return risk

def calc_risk_full(k, m, C, y, lab, unlab, m_probs, gamma2):
    N = C.shape[0]
    m_at_k = m_probs[k]
    m_k_p1 = calc_next_m(m, C, y, lab, k, 1., gamma2)
    m_k_p1 = get_probs(m_k_p1)
    risk = (1. - m_at_k)*np.sum([min(m_k_p1[i], 1.- m_k_p1[i]) for i in range(N)])
    m_k_m1 = calc_next_m(m, C, y, lab, k, -1., gamma2)
    m_k_m1 = get_probs(m_k_m1)
    risk += m_at_k*np.sum([min(m_k_m1[i], 1.- m_k_m1[i]) for i in range(N)])
    return risk

def calc_stats(m, fid, gt_flipped, _print=False):
    stats = {}

    N = m.shape[0]
    m1 = np.where(m >= 0)[0]
    m2 = np.where(m < 0)[0]


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

def plot_iter(stats, X, k_next=-1):
    corr1 = stats['corr1']
    corr2 = stats['corr2']
    sup1 = stats['sup1']
    sup2 = stats['sup2']
    incorr1 = stats['incorr1']
    incorr2 = stats['incorr2']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    plt.scatter(X[corr1,0], X[corr1,1], marker='x', c='b', alpha=0.2)
    plt.scatter(X[incorr1,0], X[incorr1,1], marker='x', c='r', alpha=0.2)
    plt.scatter(X[corr2,0], X[corr2,1], marker='o', c='r',alpha=0.15)
    plt.scatter(X[incorr2,0], X[incorr2,1], marker='o', c='b',alpha=0.15)
    plt.scatter(X[sup1,0], X[sup1,1], marker='x', c='b', alpha=1.0)
    plt.scatter(X[sup2,0], X[sup2,1], marker='o', c='r', alpha=1.0)
    plt.axis('equal')
    plt.show()
    return

def calc_stats_multi(m, fid, gt_flipped, _print=False):
    stats = {}

    N = m.shape[0]
    classes = list(fid.keys())
    offset = min(classes)

    if offset == -1:
        m1 = np.where(m >= 0)[0]
        m2 = np.where(m < 0)[0]


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

    else:
        m_class = np.argmax(m,axis=1) + offset # get the class labelings for the current m
        m_class_ind = {c:np.where(m_class == c)[0] for c in classes}

        corr = {c : list(set(m_class_ind[c]).intersection(set(gt_flipped[c]))) for c in classes}
        incorr = {}
        for corr_class, wr_class in permutations(classes, 2):
            if corr_class not in incorr.keys():
                incorr[corr_class] = [(wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class]))))]
            else:
                incorr[corr_class].append((wr_class, list(set(m_class_ind[wr_class]).intersection(set(gt_flipped[corr_class])))))

        stats['corr'] = corr
        stats['incorr'] = incorr

        tot_correct = 0.
        for corr_c_nodes in corr.values():
            tot_correct += len(corr_c_nodes)
        acc = tot_correct/N
        error = 1. - acc


    if _print:
        print('Error = %f' % error )

    return error, stats


COLORS = ['b', 'r', 'g',  'cyan', 'k', 'y']
MARKERS = ['x', 'o', '^', '+', 'v']

def plot_iter_multi(stats, X, fid, k_next=-1):
    corr = stats['corr']
    incorr = stats['incorr']
    if type(k_next) == type([1, 2]):
        plt.scatter(X[np.ix_(k_next,[0])], X[np.ix_(k_next,[1])], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next >= 0:
        plt.scatter(X[k_next,0], X[k_next,1], marker= 's', c='y', alpha= 0.7, s=200) # plot the new points to be included
        plt.title('Dataset with Label for %s added' % str(k_next))
    elif k_next == -1:
        plt.title('Dataset with Initial Labeling')

    classes = corr.keys()
    ofs = min(classes)
    num_class = len(classes)

    if num_class > len(COLORS):
        print('Not enough colors in COLORS list, reusing colors with other markers.')


    for c, c_nodes in corr.items():
        plt.scatter(X[c_nodes,0], X[c_nodes,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=0.2)
    for corr_c, vals in incorr.items():
        incorr_c, incorr_nodes = zip(*vals)
        for l in range(len(incorr_c)):
            plt.scatter(X[incorr_nodes[l],0], X[incorr_nodes[l],1], marker=MARKERS[(corr_c-ofs)%num_class],
                                        c=COLORS[(incorr_c[l]-ofs)%num_class], alpha=0.2)
    for c, fid_c in fid.items():
        plt.scatter(X[fid_c,0], X[fid_c,1], marker=MARKERS[(c-ofs)%num_class], c=COLORS[(c-ofs)%num_class], alpha=1.0)
    plt.axis('equal')
    plt.show()
    return



def calc_orig_old(v, w, B, fid, tau, alpha, gamma2):
    print('Out of date code... probably will throw error.')
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

def calc_orig(v, w, B, fid, labeled, unlabeled, tau, alpha, gamma2):
    N = v.shape[0]


    y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
    y[fid[1]] = 1.
    y[fid[-1]] = -1.

    N_prime = len(labeled)
    w_inv = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    C_tau = v.dot((v*w_inv).T)
    C_ll = C_tau[np.ix_(labeled, labeled)]
    C_all_l = C_tau[:,labeled]
    C_ll[np.diag_indices(N_prime)] += gamma2  # directly changing C_ll
    A_inv = sla.inv(C_ll)
    Block1 = C_all_l.dot(A_inv)
    C = C_tau - Block1.dot(C_all_l.T)
    m = Block1.dot(y[labeled])
    m = np.asarray(m).flatten()
    return m, np.asarray(C), y

def calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2):
    N = v.shape[0]
    classes = fid.keys()
    num_class = len(classes)
    ofs = min(classes)

    if -1 in fid.keys():
        y = np.zeros(N)  # this will already be in the expanded size, as if (H^Ty)
        y[fid[1]] = 1.
        y[fid[-1]] = -1.
    else:
        y = np.zeros((N, num_class))  # this will already be in the expanded size, as if (H^Ty)
        for c, fid_c in fid.items():
            y[fid_c,c-ofs] = 1    # TODO: make it sparse

    N_prime = len(labeled)
    w_inv = (tau ** (2 * alpha)) * np.power(w + tau**2., -alpha)     # diagonalization of C_t,e
    C_tau = v.dot((v*w_inv).T)
    C_ll = C_tau[np.ix_(labeled, labeled)]
    C_all_l = C_tau[:,labeled]
    C_ll[np.diag_indices(N_prime)] += gamma2  # directly changing C_ll
    A_inv = sla.inv(C_ll)
    Block1 = C_all_l.dot(A_inv)
    C = C_tau - Block1.dot(C_all_l.T)
    m = Block1.dot(y[labeled])
    if -1 in fid.keys():
        m = np.asarray(m).flatten()
    else:
        m = np.asarray(m)
    return m, np.asarray(C), y


def run_next_EEM(m, C, y, lab, unlab, fid, ground_truth, gamma2, verbose=False, risk_full=False):
    tic = time.clock()
    risks = []
    m_probs = get_probs_multi(m)
    for j in unlab:
        if risk_full:
            risk_j = calc_risk_full(j, m, C, y, lab, unlab, m_probs, gamma2)
        else:
            risk_j = calc_risk(j, m, C, y, lab, unlab, m_probs, gamma2)
        heappush(risks, (risk_j, j))

    k_next_risk, k_next = heappop(risks)
    toc = time.clock()
    if verbose:
        print('Time for EEM = %f' % (toc - tic))

    # Ask "the oracle" for k_next's value, known from ground truth in Ns
    y_k_next = ground_truth[k_next]
    fid[y_k_next].append(k_next)

    m_next, C_next = calc_next_C_and_m(m, C, y, lab, k_next, y_k_next, gamma2)
    y[k_next] = y_k_next
    lab = np.array(list(lab)+ [k_next])
    unlab.remove(k_next)


    return k_next, m_next, C_next, y, lab, unlab, fid

def V_opt(C, unlabeled, gamma2):
    ips = np.array([np.inner(C[k,:], C[k,:]) for k in unlabeled]).flatten()
    v_opt = ips/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(v_opt)]
    return k_max

def Sigma_opt(C, unlabeled, gamma2):
    sums = np.sum(C[unlabeled,:], axis=1)
    sums = np.asarray(sums).flatten()**2.
    s_opt = sums/(gamma2 + np.diag(C)[unlabeled])
    k_max = unlabeled[np.argmax(s_opt)]
    return k_max

def run_next_VS(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2, method='S', batch_size=5, verbose=False):
    k_to_add = []
    C_next = C.copy()
    for i in range(batch_size):
        tic = time.clock()
        if method == 'V':
            k_next = V_opt(C_next, unlabeled, gamma2)
        elif method == 'S':
            k_next = Sigma_opt(C_next, unlabeled, gamma2)
        else:
            raiseValueError('Parameter for "method" is not valid...')
        toc = time.clock()
        if verbose:
            print('Time for %s_opt = %f' % (method, (toc - tic)))

        k_to_add.append(k_next)
        unlabeled.remove(k_next)  # we are updating unlabeled here

        # calculate update of C -- the posterior of adding k
        ck = C_next[k_next,:]
        ckk = ck[k_next]
        C_next -= (1./(gamma2 + ckk))*np.outer(ck,ck)



    # Ask "the oracle" for values of the k in k_to_add value known from ground truth
    y_ks = [ground_truth[k] for k in k_to_add]

    # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
    # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
    m_next = calc_next_m_batch(m, C, y, labeled, k_to_add, y_ks, gamma2)

    del m, C  # delete now that no longer need

    # update the observations vector y, labeled, and fid
    y[k_to_add] = y_ks
    labeled.extend(k_to_add)
    for k in k_to_add:
        fid[ground_truth[k]].append(k)


    return m_next, C_next, y, labeled, unlabeled, fid, k_to_add


def run_next_VS_multi(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2, method='S', batch_size=5, verbose=False):
    k_to_add = []
    C_next = C.copy()
    for i in range(batch_size):
        tic = time.clock()
        if method == 'V':
            k_next = V_opt(C_next, unlabeled, gamma2)
        elif method == 'S':
            k_next = Sigma_opt(C_next, unlabeled, gamma2)
        else:
            raiseValueError('Parameter for "method" is not valid...')
        toc = time.clock()
        if verbose:
            print('Time for %s_opt = %f' % (method, (toc - tic)))

        k_to_add.append(k_next)
        unlabeled.remove(k_next)  # we are updating unlabeled here

        # calculate update of C -- the posterior of adding k
        ck = C_next[k_next,:]
        ckk = ck[k_next]
        C_next -= (1./(gamma2 + ckk))*np.outer(ck,ck)


    if -1 in fid.keys():
        # Ask "the oracle" for values of the k in k_to_add value known from ground truth
        y_ks = [ground_truth[k] for k in k_to_add]

        # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
        # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
        m_next = calc_next_m_batch(m, C, y, labeled, k_to_add, y_ks, gamma2)
        # update the observations vector y, labeled, and fid
        y_next = y.copy()
        y_next[k_to_add] = y_ks
        labeled.extend(k_to_add)

    else:
        # Ask "the oracle" for values of the k in k_to_add value known from ground truth
        ofs = min(list(fid.keys()))
        class_ind_ks = [ground_truth[k]-ofs for k in k_to_add]

        # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
        # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
        m_next, y_next, labeled = calc_next_m_batch_multi(m, C, y, labeled, k_to_add, class_ind_ks, gamma2)

    del m, C, y  # delete now that no longer need

    # update fid
    for k in k_to_add:
        fid[ground_truth[k]].append(k)


    return m_next, C_next, y_next, labeled, unlabeled, fid, k_to_add



def run_test_AL_VS_multi(X, v, w, fid, ground_truth, method='S', tag2=(0.01, 1.0, 0.01), test_opts=(5, 10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        ground_truth :
        method : Either 'S' (Sigma_opt) or 'V'(V_opt)
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (batch_size, iters, verbose). Default (5, 10, False)
                (batch_size int, iters int, verbose bool)
                Note if iters % batch_size != 0 then we make iters to be a multiple of batch_size
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    batch_size, iters, verbose = test_opts

    mod = iters % batch_size
    if mod != 0:
        iters += (batch_size - mod)
    num_batches = int(iters / batch_size)


    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))

    # Initial solution - find m and C, keep track of y
    tic = time.clock()
    m, C, y = calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig_multi took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        if -1 in fid.keys():
            plot_iter(stats_obj, X, k_next=-1)
        else:
            plot_iter_multi(stats_obj, X, fid, k_next=-1)
    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # AL choices - done in a batch
    for l in range(num_batches):
        m, C, y, labeled, unlabeled, fid, k_added = run_next_VS_multi(m, C, y, labeled,
                            unlabeled, fid, ground_truth, gamma2, method, batch_size, verbose)
        error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
        ERRS.append((k_added,error))
        M[l] = m
        if verbose:
            print('Iter = %d' % (l + 1))
            if -1 in fid.keys():
                plot_iter(stats_obj, X, k_next=k_added)
            else:
                plot_iter_multi(stats_obj, X, fid, k_next=k_added)
    return ERRS, M

def run_test_AL(X, v, w, fid, ground_truth, tag2=(0.01, 1.0, 0.01), test_opts=(10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (iters, verbose). Default (10, False)
                (iters int, verbose bool)
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    iters, verbose = test_opts

    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))


    # Initial solution - find m and C, keep track of y
    B = 0
    tic = time.clock()
    m, C, y = calc_orig(v, w, B, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        plot_iter(stats_obj, X, k_next=-1)

    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # AL choices
    for l in range(iters):
        k, m, C, y, labeled, unlabeled, fid = run_next_EEM(m, C, y, labeled, unlabeled, fid, ground_truth, gamma2)
        error, stats_obj = calc_stats(m, fid, gt_flipped)
        ERRS.append((k,error))
        M[l] = m
        if verbose:
            print('Iter = %d' % (l + 1))
            plot_iter(stats_obj, X, k_next=k)
    return ERRS, M

def run_test_AL_VS(X, v, w, fid, ground_truth, method='S', tag2=(0.01, 1.0, 0.01), test_opts=(5, 10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        ground_truth :
        method : Either 'S' (Sigma_opt) or 'V'(V_opt)
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (batch_size, iters, verbose). Default (5, 10, False)
                (batch_size int, iters int, verbose bool)
                Note if iters % batch_size != 0 then we make iters to be a multiple of batch_size
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    batch_size, iters, verbose = test_opts

    mod = iters % batch_size
    if mod != 0:
        iters += (batch_size - mod)
    num_batches = int(iters / batch_size)

    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))


    # Initial solution - find m and C, keep track of y
    B = 0
    tic = time.clock()
    m, C, y = calc_orig(v, w, B, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats(m, fid, gt_flipped)
    ERRS.append(([-1],error))
    if verbose:
        print('Iter = 0')
        plot_iter(stats_obj, X, k_next=-1)

    M = {}
    M[-1] = m

    # AL choices - done in a batch
    for l in range(num_batches):
        m, C, y, labeled, unlabeled, fid, k_added = run_next_VS(m, C, y, labeled,
                            unlabeled, fid, ground_truth, gamma2, method, batch_size, verbose)

        M[l] = m
        error, stats_obj = calc_stats(m, fid, gt_flipped)
        ERRS.append((k_added, error))
        if verbose:
            print('Iter = %d' % (l + 1))
            plot_iter(stats_obj, X, k_next=k_added)


    return ERRS, M

def run_test_rand(X, v, w, fid, ground_truth, tag2=(0.01, 1.0, 0.01), test_opts=(10, False)):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (iters, verbose). Default (10, False)
                (iters int, verbose bool)
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    iters, verbose = test_opts

    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))


    # Initial solution - find m and C, keep track of y
    B = 0
    m, C, y = calc_orig(v, w, B, fid, labeled, unlabeled, tau, alpha, gamma2)

    # Calculate the error of the classification resulting from this initial solution
    ERRS_rand = []
    error, stats_obj = calc_stats(m, fid, gt_flipped)
    ERRS_rand.append((-1,error))
    if verbose:
        print('Iter = 0')
        plot_iter(stats_obj, X, k_next=-1)

    # structure to record the m vectors calculated at each iteration
    M_rand = {}
    M_rand[-1] = m

    # Rand choices
    for l in range(iters):

        k_next = np.random.choice(unlabeled,1)[0]
        y_k_next = ground_truth[k_next]
        m, C = calc_next_C_and_m(m, C, y, labeled, k_next, y_k_next, gamma2)

        y[k_next] = y_k_next
        labeled = np.array(list(labeled)+ [k_next])
        unlabeled.remove(k_next)

        M_rand[l] = m
        error, stats_obj = calc_stats(m, fid, gt_flipped)
        ERRS_rand.append((k_next, error))
        if verbose:
            print('Iter = %d' % (l + 1))
            plot_iter(stats_obj, X, k_next=k_next)
    return ERRS_rand, M_rand


def run_test_rand_multi(X, v, w, fid, ground_truth, tag2=(0.01, 1.0, 0.01), test_opts=(10, False), show_all_iters=False):
    '''
    Inputs:
        X : (N x d) data matrix with the data points as columns
        v : (N x N) eigenvectors (as columns)
        w : (N, ) eigenvalues numpy array
        fid : dictionary with fidelity indices (class_i, [i_1, i_2, ...])
        ground_truth :
        method : Either 'S' (Sigma_opt) or 'V'(V_opt)
        tag2 : tuple (tau, alpha, gamma2). Default tau = 0.01, alpha = 1.0, gamma2 = 0.00001
        test_opts : tuple (iters, verbose). Default (10, False)
                (iters int, verbose bool)
        show_all_iters : bool whether or not to calculate the error/plot at each single choice from random sampling
    '''
    N = len(ground_truth)
    tau, alpha, gamma2 = tag2
    iters, verbose = test_opts


    # Prepare datastructures for labeling uses later
    gt_flipped = {}
    indices = np.array(list(range(N)))
    labeled = set()
    for k in fid.keys():
        k_mask = indices[ground_truth ==k]
        gt_flipped[k] = k_mask
        labeled = labeled.union(set(fid[k]))
    unlabeled = sorted(list(set(indices) - labeled))
    labeled = sorted(list(labeled))

    classes = list(fid.keys())
    ofs = min(classes)

    # Initial solution - find m and C, keep track of y
    tic = time.clock()
    m, C, y = calc_orig_multi(v, w, fid, labeled, unlabeled, tau, alpha, gamma2)
    toc = time.clock()
    if verbose:
        print('calc_orig_multi took %f seconds' % (toc -tic))

    # Calculate the error of the classification resulting from this initial solution
    ERRS = []
    error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
    ERRS.append((-1,error))
    if verbose:
        print('Iter = 0')
        if -1 in fid.keys():
            plot_iter(stats_obj, X, k_next=-1)
        else:
            plot_iter_multi(stats_obj, X, fid, k_next=-1)

    # structure to record the m vectors calculated at each iteration
    M = {}
    M[-1] = m

    # rand choices - done all at once
    if show_all_iters:
        for l in range(iters):
            k_next = np.random.choice(unlabeled,1)[0]
            if -1 in fid.keys():
                y_k_next = ground_truth[k_next]
                y[k_next] = y_k_next
                m, C = calc_next_C_and_m(m, C, y, labeled, k_next, y_k_next, gamma2)
                labeled = np.array(list(labeled)+ [k_next])
            else:
                class_ind_k_next = ground_truth[k_next]-ofs
                y[k_next,class_ind_k_next] = 1.
                m, C = calc_next_C_and_m_multi(m, C, y, labeled, k_next, class_ind_k_next, gamma2)
                labeled.append(k_next)

            unlabeled.remove(k_next)

            M[l] = m
            error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
            ERRS.append((k_next, error))
            if verbose:
                print('Iter = %d' % (l + 1))
                if -1 in fid.keys():
                    plot_iter(stats_obj, X, k_next=k_next)
                else:
                    plot_iter_multi(stats_obj, X, fid, k_next=k_next)
    else:
        k_to_add = list(np.random.choice(unlabeled,iters, replace=False))
        for k in k_to_add:
            unlabeled.remove(k)

        if -1 in fid.keys():
            # Ask "the oracle" for values of the k in k_to_add value known from ground truth
            y_ks = [ground_truth[k] for k in k_to_add]
            # Do BATCH calculation now that we've queried the oracle, notice this is using the OLD C
            # just had found it was a little bit faster. could just do (1/gamma2)*C_next.dot(y[labeled_new])
            m_next = calc_next_m_batch(m, C, y, labeled, k_to_add, y_ks, gamma2)
            # update the observations vector y, labeled, and fid
            y[k_to_add] = y_ks
            labeled.extend(k_to_add)

        else:
            class_ind_ks = [ground_truth[k]-ofs for k in k_to_add]
            m, C, y, labeled = calc_next_C_and_m_batch_multi(m, C, y, labeled, k_to_add, class_ind_ks, gamma2)

        error, stats_obj = calc_stats_multi(m, fid, gt_flipped)
        ERRS.append((k_to_add,error))
        M[0] = m
        if verbose:
            if -1 in fid.keys():
                plot_iter(stats_obj, X, k_next=k_to_add)
            else:
                plot_iter_multi(stats_obj, X, fid, k_next=k_to_add)

    return ERRS, M
