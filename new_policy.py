# import VL
import simulation as simu
import numpy as np
import scipy.optimize as optim


beta = np.array([-4.16796938,  2.69578732,0,0]).reshape((2,2))
def new_policy(beta,s):
    '''
    Input: 
        beta: nA*nS
        s: nS for one patient at time t
    Output:
        policy probability for next step
    '''
    k = np.dot(s,beta.T)
    expk = np.exp(k)
    expk[-1]=1
    # print(expk)
    return expk/np.sum(expk)

def one_step(beta,s):
    prob = new_policy(beta,s)
    act_next = np.random.multinomial(1,prob,size=1)
    return act_next.ravel()

def simu_new_1d(beta,s,T,nA=2):
    nS = s.shape[0]
    Si = np.zeros((T,nS))
    Ai = np.zeros((T-1,nA))
    Si[0]=s                                # initialize state
    et = np.random.normal(0,.25,(T,nS))    # noise: T*2 matrix

    for t in range(T-1):    
        A = one_step(beta,Si[t])               # fill in the empty matrix over time
        Si[t+1,0] = (3/4)*(2*A[1]-1)*Si[t,0]+(1/4)*Si[t,0]*Si[t,1]+et[t+1,0] ## A is one-hot encoding;
        Si[t+1,1] = (3/4)*(1-2*A[1])*Si[t,1]+(1/4)*Si[t,0]*Si[t,1]+et[t+1,1] ## use A[1] as integer when nA=2
        Ai[t] = A
    return Si,Ai,et

def simu_new_2d(n,beta,s_initial,T,nA=2):
    nS = s_initial.shape[1]
    S_matrix = np.zeros((n,T,nS))                # initialize a matrix for states
    A_matrix = np.zeros((n,T-1,nA))
    e_matrix = np.zeros((n,T,nS))
    for i in range(n):
        prim_simu = simu_new_1d(beta,s_initial[i],T,nA)
        S_matrix[i] = prim_simu[0]  # fill in the empty array with data for each patient
        A_matrix[i] = prim_simu[1]
        e_matrix[i] = prim_simu[2]
        # print(S_matrix[i][-1])
    return S_matrix,A_matrix, e_matrix

n=100
T=100
ss = np.ones((n,2))
S,A1hot,E = simu_new_2d(n,beta,ss,T)
A = np.zeros((n,T-1))
for i in range(n):
        A[i,:] = A1hot[i,:,1]

print(np.mean(simu.est_util(S,A)))

