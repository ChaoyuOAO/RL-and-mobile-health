'''
Adapted from https://github.com/shengzhang37/SAVE
'''

import operator
import numpy as np
import pandas as pd
from scipy.stats import norm

from utility import *
from AGENT import *
from exp_est_pol import *

from matplotlib import pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

data = pd.read_csv("D:/dingchaoyu_study/UM/dcy/M2health/SAVE/src/HS_MRT_example_v2.csv")
action0 = data[["id", "MRT_action"]]
policy0 = data[["id", "MRT_probs"]]
reward0 = data[["id", "MRT_reward"]]
state0 = data[["id","dosage","temperature","logpresteps","sqrt.totalsteps","variation"]]#,"other.location","variation",
## reason: too many knots for binary data and internal knots are required; maybe reduce degree of basis
## scale dosage
state0["dosage"] = (state0["dosage"].copy()-state0["dosage"].mean())/state0["dosage"].std()
# calculate the correlation matrix
corr = state0[["dosage","temperature","logpresteps","sqrt.totalsteps","variation", "engagement"]].corr()
print(corr)
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()

def data_trans(df0,dim="single"):
    df0 = np.array(df0)
    n = np.unique(df0[:,0])
    if dim == "multi":
        df = {int(i-1): df0[df0[:,0]==i,1:] for i in n}
    elif dim == "single":
        df = {int(i-1): df0[df0[:,0]==i,1] for i in n}
    return df

action = data_trans(action0)
policy = data_trans(policy0)
reward = data_trans(reward0)
state = data_trans(state0,dim = "multi")
for k in [12,18,33]:
    action.pop(k)
    policy.pop(k)
    reward.pop(k)
    state.pop(k)
id_new = range(37)
action_new = dict(zip(id_new,list(action.values()))) 
policy_new = dict(zip(id_new,list(policy.values()))) 
reward_new = dict(zip(id_new,list(reward.values()))) 
state_new = dict(zip(id_new,list(state.values()))) 
# print([sum(state_new[i][:,4])/450 for i in range(37)])
# def find_valid_init(S):
#     for i in range(len(S)):
#         if S[i][1] > 0:
#             break
#     return i

# # """
# Heartstep data:
# patient: 40
# time-step: 5 decision point per day; 90 days; 450 time points
# """


def main_realdata(patient = 0, error_bound = 0.01, terminate_bound = 15, alpha = 0.05, cutoff = 0.1, product_tensor = False, Lasso = False, reward_dicount = 0.5): 
    global state_new
    global action_new
    global reward_new
    S = state_new
    A = action_new
    Y = reward_new

    #### choose the initial point and get initial state (S_init) and observed value (true_value)
    # init_time_point = int(S_init_time/time_interval) ## init_time_point means the S_init_time's corresponding 
    # S_init = S[patient][init_time_point]
    S_init = S[patient][0]
    discount = [reward_dicount **(i) for i in range(len(Y[patient]))]
    true_value = np.sum(list(map(operator.mul, Y[patient], discount)))
    result_V, result_sigma = [], []
    for i in range(6):
        cut_point = 50
        S[i] = S[i][cut_point : ]
        Y[i] = Y[i][cut_point : ]
        A[i] = A[i][cut_point : ]
    T = min(S[i].shape[0] for i in range(37))  ## only consider the min(T_patients)
    print("total_T", T)
    n = 37 ## 37 patients
    beta = 3/7 ## tuning parameter for number of basis
    n_min = 6 ## 6 patient per block
    T_min = 80
    env = setting(T = T, dim = 5)
    a = simulation(env, n = n, product_tensor = product_tensor, reward_dicount = reward_dicount) ## control the product tensor
    #a.gamma = 0.9 ## choose longer tail?
    L = int(np.sqrt((n * T_min) ** beta))
    print("L: ", max(7,(L + 3)), "d: ", 3)
    print("number of basis: ", L)
    K_n = n // n_min
    K_T = T // T_min
    print("K_n: ", K_n,"\nK_T: ", K_T)
    a.buffer_next_block(n_min, T_min, T, n = None) ## get the shape for updating
    ## replace the next block (simulated data) by the real data
    next_block = {}
    flag_exp=0
    ## start with block (1,1), patient_id:(0,5), time:(0:91)
    for i in range(n_min):
        next_block[i] = [list(S[i][0:(T_min+1)]), list(A[i][0:T_min]), list(Y[i][0:T_min]), len(list(A[i][0:T_min]))]
    a.next_block = next_block
    for rep in range(K_n * K_T):
        a.append_next_block_to_buffer()
        if product_tensor:
            a.B_spline(L = 5, d = 2)
        else:
            a.B_spline(L = max(7,(L + 3)), d = 3)
        error = 1
        terminate_loop = 0
        while error > error_bound and terminate_loop < terminate_bound:
            a._stretch_para()
            tmp = a.all_para
            a.update_op(Lasso = Lasso)
            a._stretch_para()
            error = np.sqrt(np.mean((tmp - a.all_para)**2))
            if error>1000:
                flag_exp=rep
                print("a_para: ",a.para)
            terminate_loop += 1
            print("in k = %d, terminate_loop %d, error is %.3f" %(rep, terminate_loop, error))
        a.buffer_next_block(n_min, T_min, T, n = None)
        print("current",a.current_block_idx,"next",a.next_block_idx)
        ## store sigma and V
        if a.next_block:
            tt = a.next_block_idx[1]
            print("tt",tt)
            next_block = {}
            start_i, end_i, T_block =  get_idx_pos(a.next_block_idx, n, T, n_min, T_min)
            print("start",start_i,"end",end_i, "T_block",T_block)
            for i in range(start_i,end_i):  
                next_block[i] = [list(S[i][((tt-1) * T_min) : (tt * T_min+1)]), list(A[i][((tt-1) * T_min): (tt * T_min)]), list(Y[i][((tt-1) * T_min) : (tt * T_min)]), T_block]

            a.next_block = next_block
            a._sigma(a.opt_policy, S_init, block = True)
            V = a.V(S_init, a.opt_policy)
            print("current index is (%d, %d), length of current buffer %d , length of first one %d, value is %.2f, sigma2 is %.2f "%(a.current_block_idx[0], a.current_block_idx[1], len(a.buffer), a.buffer[0][3], V, a.sigma2))
            result_V.append(V)
            result_sigma.append(float(np.sqrt(a.sigma2)))


    print("dimension of basis spline", a.para_dim)
    print(result_V)
    print(result_sigma)
    K = len(result_sigma) + 1
    V_tilde = np.sum([result_V[i] / result_sigma[i] for i in range(K - 1)]) /  np.sum([1/ result_sigma[i] for i in range(K - 1)])
    sigma_tilde = (K - 1) / np.sum([1/ result_sigma[i] for i in range(K - 1)])

    lower_bound = V_tilde - norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
    upper_bound = V_tilde + norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
    if upper_bound > true_value:
        useful = 1
    else:
        useful = 0
    f = open("Real_data_cutoff_%.2f_Lasso_%d_product_tensor_%d_reward_dicount_%.2f.txt" % (cutoff, int(Lasso), int(product_tensor), reward_dicount), "a+")
    f.write("## sklearn, dim 5 (engage), lasso=false, burn-in: 50 time steps\n")
    if flag_exp>0:
        f.write("Problems at k=%d\n"%flag_exp)
    f.write("For patient %d, v_tilde is %.3f, sigma_tilde is %.3f, lower_bound is %.3f, upper bound is %.3f, true_value is %.3f \r\n useful : %d \r\n" % (patient, V_tilde, sigma_tilde, lower_bound, upper_bound, true_value, useful))
    f.close()
    # return list(V_tilde, lower_bound, upper_bound)


main_realdata(patient = 0, error_bound = 0.01,reward_dicount = 0.5)
main_realdata(patient = 0, error_bound = 0.01,reward_dicount = 0.7)

def plot_opt(discount_rate,upper1,lower1,upper2, lower2,title):
    global state_new
    global action_new
    global reward_new
    discount = [discount_rate **(i) for i in range(450)]
    patient_id = range(0,37)
    true_vals = [np.sum(list(map(operator.mul, reward_new[patient], discount))) for patient in range(37)]
    v_tilde1 = (upper1+lower1)/2
    v_tilde2 = (upper2+lower2)/2
    print("opt_rate 1: ",sum([true_vals[i]<upper1 for i in range(len(true_vals))])/len(true_vals))
    print("opt_rate 2: ",sum([true_vals[i]<upper2 for i in range(len(true_vals))])/len(true_vals))
    ## 0.7297
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.scatter(patient_id,[true_vals[i] for i in range(len(true_vals))])
    plt.title(title)
    plt.axhline(y=lower1, color='r', linestyle='--')
    plt.axhline(y=upper1, color='r', linestyle='--')
    plt.axhline(y=lower2, color='b', linestyle='--')
    plt.axhline(y=upper2, color='b', linestyle='--')
    plt.show()
