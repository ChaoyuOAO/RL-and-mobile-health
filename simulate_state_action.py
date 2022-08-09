import numpy as np

def simu1d(T,An,Rp=2):
    # simulating states for one patient
    # Arguements:
    # T: total time points 
    # An: actions for each time point (1*T)
    # Rp: dimension of states (2 by default)

    # Output: a tuple for states
    et = np.random.normal(0,.25,(Rp,T))    # noise: 2*T matrix
    Si = np.zeros((Rp,T))                  # initialize state
    Si[0,0] = et[0,0]                      # assume at t=0, S01=S02=0
    Si[1,0] = et[1,0]
    for t in range(T-1):                   # fill in the empty matrix over time
        Si[0,t+1] = (3/4)*(2*An[t]-1)*Si[0,t]+(1/4)*Si[0,t]*Si[1,t]+et[0,t+1]
        Si[1,t+1] = (3/4)*(1-2*An[t])*Si[1,t]+(1/4)*Si[0,t]*Si[1,t]+et[1,t+1]
    return Si


def simu2d(n,T,p=.5,Rp=2,ranseed = 1):
    # Simulating states and actions for n patients over T time points
    # Arguements:
    # n: number of patients
    # T: total time points
    # p: prob for providing treatment (A_i=1)
    # Output: a tuple for a n*Rp*T array (patient #, state #, time point) for states 
    #         a n*T matrix for actions

    np.random.seed(ranseed)
    AT_n = np.random.binomial(1,p,(n,T-1))       # n*(T-1) matrix
    S_matrix = np.zeros((n,Rp,T))                # initialize a matrix for states
    for i in range(n):
        S_matrix[i] = simu1d(T, AT_n[i],Rp=2)    # fill in the empty array with data for each patient
    return S_matrix,AT_n

def est_util(S_matrix, A):
    # Estimating utility with known state and action sequence
    # Arguements:
    # S_matrix: a n*Rp*T array (patient #, state #, time point) for states
    # A: a n*(T-1) matrix for actions
    # Output: a n*(T-1) matrix for utility
    n,_,T = S_matrix.shape
    UT_n = np.zeros((n,T-1))
    for i in range(n):
        for j in range(T-1):
            UT_n[i,j] = 2*S_matrix[i,0,j+1]+S_matrix[i,1,j+1]-(1/4)*(2*A[i,j]-1)
    return UT_n

## simulate table 
## n=25
## n=25, T=24
n25_T24 = simu2d(25,24)
st_2524, act_2524 = n25_T24
util_2524 = est_util(st_2524,act_2524)
## n=25, T=36
n25_T36 = simu2d(25,36)
st_2536, act_2536 = n25_T36
util_2536 = est_util(st_2536,act_2536)
## n=25,T=48
n25_T48 = simu2d(25,48)
st_2548, act_2548 = n25_T48
util_2548 = est_util(st_2548,act_2548)

print(st_2524,"\n",act_2524,"\n",util_2524)
print(st_2536,"\n",act_2536,"\n",util_2536)
print(st_2548,"\n",act_2548,"\n",util_2548)








