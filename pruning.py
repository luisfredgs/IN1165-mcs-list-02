import numpy as np
from copy import deepcopy
import gc
gc.enable()

DTY_FLT = 'float32'
DTY_INT = 'int32'
DTY_BOL = 'bool'
CONST_ZERO = 1e-16

GAP_INF = 2 ** 31 - 1
GAP_MID = 1e8
GAP_NAN = 1e-16

def check_zero(temp):
    return temp if temp != 0. else CONST_ZERO



# ----------  Probability of Discrete Variable  -----------


# probability of one vector
#
def prob(X):
    X = np.array(X)
    vX = np.unique(X).tolist()
    dX = len(vX)
    px = np.zeros(dX)
    for i in range(dX):
        px[i] = np.mean(X == vX[i])
    px = px.tolist()
    del i, X, dX
    gc.collect()
    return deepcopy(px), deepcopy(vX)  # list


# joint probability of two vectors
#
def jointProb(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    vX = np.unique(X).tolist()
    vY = np.unique(Y).tolist()
    dX = len(vX)
    dY = len(vY)
    pxy = np.zeros((dX, dY))
    for i in range(dX):
        for j in range(dY):
            pxy[i, j] = np.mean((X == vX[i]) & (Y == vY[j]))
    pxy = pxy.tolist()
    del dX, dY, i, j, X, Y
    gc.collect()
    return deepcopy(pxy), deepcopy(vX), deepcopy(vY)  # list


# ----------  Shannon Entropy  -----------
# calculate values of entropy
# H(.) is the entropy function and p(.,.) is the joint probability


# for a scalar value
#
def H(p):
    if p == 0.:
        return 0.
    return (-1.) * p * np.log2(p)


# H(X), H(Y) :  for one vector
#
def H1(X):
    px, _ = prob(X)
    # calc
    ans = 0.
    for i in px:
        ans += H(i)

    i = -1
    del px, i
    gc.collect()
    return ans


# H(X,Y) :  for two vectors
#
def H2(X, Y):
    pxy, _, _ = jointProb(X, Y)
    # calc
    ans = 0.
    for i in pxy:
        for j in i:
            ans += H(j)

    i = j = -1
    del pxy, i, j
    gc.collect()
    return ans



# I(.;.) is the mutual information function
# I(X; Y)
#
def I(X, Y):
    px, _ = prob(X);    py, _ = prob(Y)
    pxy, _, _ = jointProb(X, Y)

    # calc
    ans = 0.
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i][j] == 0.:
                ans += 0.
            else:
                ans += pxy[i][j] * np.log2( pxy[i][j] / px[i] / py[j] )

    i = j = -1
    del px,py,pxy, i,j
    gc.collect()
    return ans




# MI(X, Y): The normalized mutual information of two discrete random variables X and Y
#
def MI(X, Y):
    tem = np.sqrt(H1(X) * H1(Y))
    ans = I(X, Y) / check_zero(tem)
    return ans

# VI(X, Y): the normalized variation of information of two discrete random variables X and Y
#
def VI(X, Y):
    return 1. - I(X, Y) / check_zero(H2(X, Y))

# For two feature vectors like p and q, and the class label vector L, define TDAC(p,q) as follows:
#
def TDAC(X, Y, L, lam):  # lambda
    if X == Y:  # list
        return 0.
    return lam * VI(X, Y) + (1. - lam) * (MI(X, L) + MI(Y, L)) / 2.

def tdac_sum(p, S, L, lam):
    S = np.array(S);    n = S.shape[1]
    # calc
    ans = 0.
    for i in range(n):
        ans += TDAC(p, S[:, i].tolist(), L, lam)
    del S,n,i
    gc.collect()
    return ans


# T is the set of individuals; S = [True,False] represents this one is in S or not, and S is the selected individuals.
#
def arg_max_p(T, S, L, lam):
    T = np.array(T);    S = np.array(S)

    # calc
    all_q_in_S = T[:,S].tolist()
    idx_p_not_S = np.where(np.logical_not(S))[0]
    if len(idx_p_not_S) == 0:
        del T,S, all_q_in_S, idx_p_not_S
        return -1  # idx = -1

    ans = [ tdac_sum(T[:,i].tolist(), all_q_in_S, L, lam)  for i in idx_p_not_S]
    idx_p = ans.index( np.max(ans) )
    idx = idx_p_not_S[idx_p]

    del T,S, all_q_in_S, idx_p_not_S, idx_p, ans
    gc.collect()
    return idx

def COMEP(T, k, L, lam):
    T = np.array(T);    n = T.shape[1]
    S = np.zeros(n, dtype=DTY_BOL)
    p = np.random.randint(0, n)
    S[p] = True
    for _ in range(1, k):
        idx = arg_max_p(T, S, L, lam)
        if idx > -1:
            S[idx] = True  #1
    S = S.tolist()
    del T,n, p, #i,idx
    gc.collect()
    # return copy.deepcopy(S)  #S  #list
    return deepcopy(S)


def COMEP_Pruning(T, k, L, lam):
    P = COMEP(T, k, L, lam)
    P = np.where(P)[0].tolist()
    return deepcopy(P)
