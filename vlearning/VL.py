import scipy.optimize as optim
import numpy as np
import matplotlib.pyplot as plt
import simulation as simu
import scipy.linalg as la

def a1hot(a,nA):
    '''
    a: nRep*T
    output: nRep*T*nA onehot coding
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
        s: n*(T+1)*nS matrix for all patients
        (binary)beta: 1*nS matrix: one row for one action under each state// (nA-1)*nS
    -------------------------------------------------------
    Output: 
        pi features: n*T*nA
    '''
    n,T,_ = s.shape
    nA = beta.shape[0]
    pif = np.zeros((n,T-1,nA))
    for i in range(n):
        k = np.vstack((np.dot(s[i][:-1],beta),np.zeros(T-1))).T
        expk = np.exp(k)
        pif[i] = expk/expk.sum(axis=1)[:,None]
    # for i in range(n):
    #     k = np.vstack((np.dot(s[i][:-1],beta.T),np.zeros(T-1))).T
    #     max_ = np.max(k)
    #     expk = np.exp(k - max_)
    #     pif[i] = expk/expk.sum(axis=1)[:,None]
    # exp_ = np.exp(dots - max_)  #Subtract this in exponent to avoid overflow
    # return expk/ np.sum(exp_) 
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
    Policy probability of Action chosen n*(T-1)

    '''
    pi_list = pi_sbeta(s,beta)
    n,T,nA = pi_list.shape
    p_a = np.zeros((n,T))
    for i in range(n):
        p_a[i] = np.sum(np.multiply(a_1hot[i],pi_list[i]),axis=1)
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
    if method == "polynomial":
        s1 = np.ndarray.flatten(s[:,:,0]).reshape((n,T,1))
        s2 = np.ndarray.flatten(s[:,:,1]).reshape((n,T,1))

        return np.concatenate((np.ones((n,T,1)),s,np.multiply(s1,s2),\
            np.multiply(s1,s1),np.multiply(s2,s2)),axis=2)
    
def get_M(vf,gamma=0.9):
    '''
    Input: 
        vf: value features (psi), n*T*nV 
        gamma: discount rate, [0,1]
    -----------------------------------------
    Output: 
        M: gamma*psi(t)Xpsi(t+1)-psi(t)Xpsi(t), n*nV*nV
    '''
    n,T,nV = vf.shape
    M0 = np.zeros((n,T-1,nV,nV))
    for i in range(n):
        for t in range(T-1):
            outer_it = gamma * np.outer(vf[i][t], vf[i][t+1])-np.outer(vf[i][t], vf[i][t]) 
            # sM_i,t
            M0[i][t] = outer_it
    return M0


def obj_func(beta, policy, M_list, A_list, S_list,R_list, vf_list, Mu_list, eps=0.01):
    '''
    Input:
        beta: nA*nS vector
        policy: nS*nA matrix: pi(ai|si) (only for the action chosen at time t)
        M_list: n*(T-1)*nV*nV: psi(t)Xpsi(t)-gamma*psi(t)Xpsi(t+1)
        A_list: n*(T-1)*nA: action matrix list (one-hot encoding)
        R_list: n*(T-1): utility list
        vf_list: n*T*nV: value feature list
        Mu_list: n*(T_1)*nS: known randomization probability
    ------------------------------------------
    Output:
        sum_w_M : sum_T pi/mu * Mt (nV x nV-size array) 
        sum_w_RS : sum_t pi/mu reward * psi(t) (nV-size array)
    --------------------------------------------
    '''
    n,T,_ = A_list.shape
    nV = vf_list.shape[2]
    sum_w_RS = np.zeros(nV) 
    sum_w_M = np.zeros((nV,nV))
    prob = policy(A_list,S_list,beta,eps)
    for i in range(n):
        w = prob[i]/Mu_list[i]
        ## pi/mu for each time
        sum_w_RS += np.sum(np.multiply(np.multiply(w, R_list[i])[:,None],vf_list[i,:-1,:]), axis=0)
        sum_w_M += np.sum(np.multiply(w[:,None,None],M_list[i]), axis=0)
    return sum_w_M/n,sum_w_RS/n
    # L = np.dot(sum_w_M/n,theta)+sum_w_RS/n
    # return np.dot(L.T,L)+l*np.dot(theta.T,theta)

def theta_opt(beta,policy, M_list, A_list, S_list, R_list, vf_list, Mu_list,eps=0,l=0.1):
    # objective = lambda theta: obj_func(beta, theta, policy, M_list, A_list, S_list,R_list, vf_list, Mu_list, eps,l)
    # opt = optim.minimize(objective, x0=theta0, method='BFGS')  
    # return opt.x
    # sum_w_M, sum_w_RS = obj_func(beta, policy,M_list, A_list, S_list,R_list, vf_list, Mu_list,eps)
    # nV = len(sum_w_RS)
    # LU = la.lu_factor(np.dot(sum_w_M,sum_w_M.T) + l*np.eye(nV)) 
    # return la.lu_solve(LU, -np.dot(sum_w_M.T,sum_w_RS))
    mean_M, mean_RS = obj_func(beta, policy,M_list, A_list, S_list,R_list, vf_list, Mu_list,eps)
    nV = len(mean_RS)
    A = np.dot(mean_M,mean_M.T) + l*np.eye(nV)
    b = -np.dot(mean_M.T,mean_RS)
    thetaOpt = np.linalg.solve(A,b)

    return thetaOpt


def Vpi(beta, policy, M_list, A_list, S_list, R_list, vf_list, Mu_list,eps=0.01,l=0.1):
    '''
    Input:
        vf_list: value features (psi): n*T*nV
        theta: parameters for value: nV
    Output:
        value function: n*T
    '''
    # global betaplt
    # print(beta)
    # betaplt.plot(beta[0],beta[1], '.r-')
    theta_hat = theta_opt(beta, policy, M_list, A_list, S_list, R_list, vf_list, Mu_list,eps,l)
    # print(theta_hat)
    v_list = np.dot(vf_list,theta_hat)
    # print(v_list)
    return -np.mean(v_list)

def beta_opt(beta0,policy, M_list, A_list, S_list,R_list, vf_list, Mu_list, eps=0.01,l=0.1):
    '''
    Optimizes policy value over class of softmax policies indexed by beta. 
    Dictionary {'betaHat':estimate of beta, 'thetaHat':estimate of theta, 'objective':objective function (of policy parameters)}
    '''
    objective = lambda beta: Vpi(beta, policy, M_list, A_list, S_list, R_list, vf_list, Mu_list,eps,l)
    betaOpt = optim.minimize(objective, x0=beta0, method='L-BFGS-B').x
    thetaOpt = theta_opt(betaOpt,policy, M_list, A_list, S_list, R_list, vf_list, Mu_list,eps,l)
    return {'betaHat':betaOpt, 'thetaHat':thetaOpt}
