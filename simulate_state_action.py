import numpy as np

def simu1d(T,An,Rp=2):
    # simulating states for one patient
    # Arguements:
    # T: total time points 
    # An: actions for each time point (1*T)
    # Rp: dimension of states (2 by default)

    # Output: a tuple for states
    et = np.random.normal(0,.25,(T,Rp))    # noise: T*2 matrix
    Si = np.zeros((T,Rp))                  # initialize state
    Si[0,0] = 1+et[0,0]                    # assume at t=0, S01=S02=0
    Si[0,1] = 1+et[0,1]
    for t in range(T-1):                   # fill in the empty matrix over time
        Si[t+1,0] = (3/4)*(2*An[t]-1)*Si[t,0]+(1/4)*Si[t,0]*Si[t,1]+et[t+1,0]
        Si[t+1,1] = (3/4)*(1-2*An[t])*Si[t,1]+(1/4)*Si[t,0]*Si[t,1]+et[t+1,1]
    return Si,et


def simu2d(n,T,p=.5,Rp=2,ranseed=1):
    # Simulating states and actions for n patients over T time points
    # Arguements:
    # n: number of patients
    # T: total time points
    # p: prob for providing treatment (A_i=1)
    # Output: a tuple for a n*T*Rp array (patient #, state #, time point) for states 
    #         a n*T matrix for actions

    np.random.seed(ranseed)
    AT_n = np.random.binomial(1,p,(n,T-1))       # n*(T-1) matrix
    S_matrix = np.zeros((n,T,Rp))                # initialize a matrix for states
    e_matrix = np.zeros((n,T,Rp))
    for i in range(n):
        prim_simu = simu1d(T, AT_n[i],Rp)
        S_matrix[i] = prim_simu[0]  # fill in the empty array with data for each patient
        e_matrix[i] = prim_simu[1]
    return S_matrix,AT_n, e_matrix

def est_util(S_matrix, A):
    # Estimating utility with known state and action sequence
    # Arguements:
    # S_matrix: a n*Rp*T array (patient #, state #, time point) for states
    # A: a n*(T-1) matrix for actions
    # Output: a n*(T-1) matrix for utility
    n,T,_ = S_matrix.shape
    UT_n = np.zeros((n,T-1))
    for i in range(n):
        for j in range(T-1):
            UT_n[i,j] = 2*S_matrix[i,j+1,0]+S_matrix[i,j+1,1]-(1/4)*(2*A[i,j]-1)
    return UT_n

def stable_simu(n,T,p=.5,Rp=2):
    S_matrix0,An0,et_matirx0 = simu2d(n,50+T,p,Rp)
    S_matrix = S_matrix0[:,50:,:]
    An = An0[:,50:]
    return S_matrix,An,et_matirx0

def start(s10,s20,T,A):
    Si = np.zeros((T,2))                          # initialize state
    Si[0,0] = s10                                 # assume at t=0, S01=S02=0
    Si[0,1] = s20
    for t in range(T-1):                          # fill in the empty matrix over time
        Si[t+1,0] = (3/4)*(2*A[t]-1)*Si[t,0]+(1/4)*Si[t,0]*Si[t,1]
        Si[t+1,1] = (3/4)*(1-2*A[t])*Si[t,1]+(1/4)*Si[t,1]*Si[t,1]
    return Si

def multi_start(n,s10,s20,T,Rp = 2, p=.5,ranseed = 1):
    np.random.seed(ranseed)
    AT_n = np.random.binomial(1,p,(n,T-1))         # n*(T-1) matrix
    S_matrix = np.zeros((n,T,Rp))                  # initialize a matrix for states
    for i in range(n):
        S_matrix[i] = start(s10,s20,T, AT_n[i])    # fill in the empty array with data for each patient
    return S_matrix,AT_n







