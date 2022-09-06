import scipy.optimize as optim
import numpy as np
import simulation as simu
import scipy.linalg as la

def a1hot(a,nA):
    '''
    a: nRep*(T-1)
    output: nRep*(T-1)*nA onehot coding
    '''
    nRep,T = a.shape
    a_mtx = np.zeros((nRep,T,nA))
    for i in range(nRep):
        for j in range(T):
            a_mtx[i][j]=np.array([a[i][j]==0,a[i][j]==1])
    return a_mtx

def pi_sbeta(s,beta):
    '''
    Input:
        s: n*T*nS matrix for one patient
        beta: nA*nS matrix: one row for one action under each state
    -------------------------------------------------------
    Output: 
        pi features: n*T*nA
    '''
    n,T,_ = s.shape
    nA = beta.shape[0]
    pif = np.zeros((n,T,nA))
    for i in range(n):
        k = np.dot(s[i],beta.T)
        expk = np.exp(k)
        pif[i] = expk/expk.sum(axis=1)[:,None]
    return pif

def policy(a_1hot,s,beta,eps=0.01):
    '''
    Input:
    a_1hot: one-hot encoding for action matrix: n*(T-1)*nA
    s: state matrix: n*T*nS
    beta: parameters for each state with different actions: nA*nS
    eps: epsilon greedy algorithm
    -------------------------------------------------------------
    Output:

    '''
    pi_list = pi_sbeta(s,beta)
    n,T,nA = pi_list.shape
    p_a = np.zeros((n,T-1))
    for i in range(n):
        p_a[i] = np.diag(np.dot(a_1hot[i,:,:],pi_list[i,:-1,:].T))
    return p_a*(1-eps)+eps/nA

def Vfeatures(s,method="linear"):
    '''
    Input: state matrix, basis function
    ----------------------------------
    Value function is psi^T theta^pi
    Output: psi: n*T*nV
        linear: n*T*(nS+1) (intercept included)

    '''
    n,T,nS = s.shape
    if method == "linear":
        return np.concatenate((np.ones((n,T,1)),s),axis=2)
    
def get_M(vf,gamma=0.1):
    '''
    Input: 
        vf: value features (psi), n*T*nV 
        gamma: discount rate, [0,1]
    -----------------------------------------
    Output: 
        M: gamma*psi(t)Xpsi(t+1)-psi(t)Xpsi(t), n*nV*nV
    '''
    n,T,nV = vf.shape
    M0 = np.zeros((n,nV,nV))
    for i in range(n):
        for t in range(T-1):
            outer_it = gamma * np.outer(vf[i][t], vf[i][t+1])-np.outer(vf[i][t], vf[i][t]) 
            # sum_T M_i,t
            M0[i] = M0[i]+outer_it
            # sum_n sum_T M_i,t
    return M0


def LAMBDA_components(beta, policy, M_list, A_list, S_list,R_list, vf_list, Mu_list, eps=0.01):
  '''
  Input:
    beta: nA*nS vector
    policy: nS*nA matrix: pi(ai|si) (only for the action chosen at time t)
    M_list: n*nV*nV: psi(t)Xpsi(t)-gamma*psi(t)Xpsi(t+1)
    A_list: n*T*nA: action matrix list (one-hot encoding)
    R_list: n*T: utility list
    vf_list: n*T*nV: value feature list
    Mu_list: n*T*nS: known randomization probability
  ------------------------------------------
  Output:
    sum_w_M : sum_T pi/mu * Mt (nV x nV-size array) 
    sum_w_RS : sum_t pi/mu reward * psi(t) (nV-size array)
  '''
  n,T,_ = A_list.shape
  nV = vf_list.shape[2]
  sum_w_RS = np.zeros(nV) 
  sum_w_M = np.zeros((nV,nV))
  for i in range(n):
    w = policy(A_list,S_list,beta,eps)[i]/Mu_list[i]
    ## pi/mu for each time
    sum_w_RS += np.sum(np.multiply(np.multiply(w, R_list[i]).reshape(T,1),vf_list[i][:-1,:]), axis=0)
    sum_w_M += np.sum(np.multiply(M_list[i], w.reshape(T,1,1)), axis=0)
  return sum_w_M, sum_w_RS

def LAMBDA(sum_w_M,sum_w_RS,theta,n):
    '''
    LAMBDA value
    '''
    return np.dot(sum_w_M/n,theta)+sum_w_RS/n

def thetaPi_obj0(beta, policy, M_list, A_list, S_list, R_list, fv_list, Mu_list,eps=0.01):  
  '''
  Estimate of theta: min(LAMBDA+lambda*theta)
  '''
  sum_w_M, sum_w_RS = LAMBDA_components(beta, policy,M_list, A_list, S_list,R_list, fv_list, Mu_list,eps)
  nV = len(sum_w_RS)
  LU = la.lu_factor(sum_w_M + 0.01*np.eye(nV)) 
  return la.lu_solve(LU, -sum_w_RS)

def obj_func(Lambda,theta,l):
    return np.dot(Lambda.T,Lambda)+l*np.dot(theta.T,theta)

def value_func(fv_list,theta):
    '''
    Input:
        fv_list: value features (psi): n*T*nV
        theta: parameters for value: nV
    Output:
        value function: n*T
    '''

start11 = simu.multi_start(25,1,1,24)
s11, a11 = start11
util_11 = simu.est_util(s11,a11)
a11hot = a1hot(a11,2)
# set beta to be 0
beta11 = np.array([0.5,-0.2,-0.1,0.6]).reshape((2,2))
pi11 = pi_sbeta(s11,beta11)
vf11 = Vfeatures(s11)
print(vf11.shape)
M11 = get_M(vf11)
Mu11 = np.array([0.5]*23*25).reshape(25,23)
sum1,sum2 = LAMBDA_components(beta11,policy,M11,a11hot,s11,util_11,vf11,Mu11,0.01)
ll = LAMBDA(sum1,sum2,np.array([1,2,3]),25)
theta_obj1 = thetaPi_obj0(beta11,policy,M11,a11hot,s11,util_11,vf11,Mu11,0.01)
print(theta_obj1)
print(obj_func(ll,np.array([-0.00223447,0.00824631,0.00452626]),0.01))
