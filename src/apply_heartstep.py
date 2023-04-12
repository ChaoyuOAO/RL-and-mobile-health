import operator
import numpy as np


from scipy.stats import norm

import numpy as np
import operator
import pandas as pd

import operator

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
# corr = state0[["dosage","temperature","logpresteps","sqrt.totalsteps","variation", "engagement"]].corr()
# print(corr)
# # plot the heatmap
# sns.heatmap(corr, 
#         xticklabels=corr.columns,
#         yticklabels=corr.columns)
# plt.show()

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
    """
    Input:
        patient = 0 ~ 37 represent different patient
        error_bound: stop error bound for double fitted q learning
        terminate_bound: iteration bound for double fitted q learning
        alpha: significance level
        cutoff : > cutoff means valid action
        product_tensor: if True we use product bspline, otherwise, we use additive bslpline
        Lasso: if True, use Lasso loss in  double fitted q learning update
        reward_dicount: reward discount decay rate
    """
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
    # for i in range(6):
    #     cut_point = find_valid_init(S[i])
    #     S[i] = S[i][cut_point : ]
    #     Y[i] = Y[i][cut_point : ]
    #     A[i] = A[i][cut_point : ]
    T = min(S[i].shape[0] for i in range(37))  ## only consider the min(T_patients)
    n = 37 ## 37 patients
    beta = 3/7 ## tuning parameter for number of basis
    n_min = 6 ## 6 patient per block
    T_min = 90
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
        a._sigma(a.opt_policy, S_init, block = True)
        V = a.V(S_init, a.opt_policy)
        ## store sigma and V
        
        if a.next_block_idx:
            tt = a.next_block_idx[1]
            next_block = {}
            start_i, end_i, T_block =  get_idx_pos(a.next_block_idx, n, T, n_min, T_min)
            print("start",start_i,"end",end_i)
            for i in range(start_i,end_i):  
                next_block[i] = [list(S[i][((tt-1) * T_min) : (tt * T_min+1)]), list(A[i][((tt-1) * T_min): (tt * T_min)]), list(Y[i][((tt-1) * T_min) : (tt * T_min)]), T_block]

            a.next_block = next_block
                    ## store sigma and V
            print("current index is (%d, %d), length of current buffer %d , length of first one %d, value is %.2f, sigma2 is %.2f "%(a.current_block_idx[0], a.current_block_idx[1], len(a.buffer), a.buffer[0][3], V, a.sigma2))
            result_V.append(V)
            result_sigma.append(np.sqrt(a.sigma2))

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
    if flag_exp>0:
        f.write("Problems at k=%d"%flag_exp)
    f.write("For patient %d, lower_bound is %.3f, upper bound is %.3f, true_value is %.3f \r\n useful : %d \r\n" % (patient, lower_bound, upper_bound, true_value, useful))
    f.close()
    # return list(V_tilde, lower_bound, upper_bound)


main_realdata(patient = 0, error_bound = 0.01,reward_dicount = 0.5)
# main_realdata(patient = 0, error_bound = 0.01,reward_dicount = 0.7)

# def plot_opt(discount_rate,upper1,lower1,upper2, lower2,title):
#     global state_new
#     global action_new
#     global reward_new
#     discount = [discount_rate **(i) for i in range(450)]
#     patient_id = range(0,37)
#     true_vals = [np.sum(list(map(operator.mul, reward_new[patient], discount))) for patient in range(37)]
#     v_tilde1 = (upper1+lower1)/2
#     v_tilde2 = (upper2+lower2)/2
#     print("opt_rate 1: ",sum([true_vals[i]<upper1 for i in range(len(true_vals))])/len(true_vals))
#     print("opt_rate 2: ",sum([true_vals[i]<upper2 for i in range(len(true_vals))])/len(true_vals))
#     ## 0.7297
#     plt.rcParams["figure.figsize"] = [7.50, 3.50]
#     plt.rcParams["figure.autolayout"] = True
#     plt.scatter(patient_id,[true_vals[i] for i in range(len(true_vals))])
#     plt.title(title)
#     plt.axhline(y=lower1, color='r', linestyle='--')
#     plt.axhline(y=upper1, color='r', linestyle='--')
#     plt.axhline(y=lower2, color='b', linestyle='--')
#     plt.axhline(y=upper2, color='b', linestyle='--')
#     plt.show()

# plot_opt(0.5,11.114,9.479, 11.303, 9.407, "Discount=0.5, p=5(r), p=6(b)")
# plot_opt(0.7,17.315,14.974,17.933, 15.402, "Discount=0.7,p=5(r), p=6(r)")
# plot_opt(0.5,11.114,9.479,11.402, 9.442, "Discount=0.5,p=5, Tmin=50, 90")
# for i in range(2):
#     plt.plot(reward_new[i])
# plt.show()
# for i in range(3):
#     main_realdata(patient = i, reward_dicount = 0.5, S_init_time =0)

# print(np.array([state[i][0] for i in (range(10))]))

sig2_05_p5_T90_sklearn = np.array([14.45491372, 14.40696118, 15.7659896, 53.33678626, 23.8959146, 149.54650758, 366.62956997, 
                                    167.84048009, 885.31073238, 76.94120942, 65.23228146, 3559.48653798, 74.17432052, 156.93116524,
                                    242.37413164, 24.53830171, 742.78605079, 694.01435279, 8949.26508156, 80.67523417, 56.55083636, 
                                    178.73974785, 100.32562219, 23.48840605, 45.90413575, 103.47034637, 108.56708414, 40.2347936,53.45555233])
# print(sig2_05_p5_T90_sklearn.shape)

# some_matrix = np.random.rand(10,10)

# cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)

# plt.matshow(some_matrix, cmap=cmap)

# plt.show()

# [array([[75.76833991]]), array([[12.73887316]]), array([[32.59449851]]), array([[55.27487011]]),
#  array([[13.51151127]]), array([[1612.66607459]]), array([[50677.86916095]]), array([[43.997096951]]),
#  array([[2741.61279257]]), array([[151.68965229]]), array([[73.95673646]]), array([[128.05318362]]), 
# array([[108.12771083]]), array([[7158.13438182]]), array([[51.587475]]), array([[174.09248227]]), 
# array([[75.22306286]]), array([[281.81115698]]), array([[53.66877566]]), array([[2216.16385549]]), 
# array([[50.43074365]]), array([[59.35848403]]), array([[262.38397175]]), array([[368.40398458]]), 
# array([[4530.44805662]]), array([[287.83233757]]), array([[292.43002635]]), array([[172.68920783]]), 
# array([[61.00265855]]), array([[37.70193999]]), array([[104.80590733]]), 
# array([[48.90539186]]), array([[80.39025924]]), array([[90.25961067]]), array([[65.64008763]]), 
# array([[35.60444649]]), array([[51.93498007]]), array([[322.10819829]]), array([[323.10876108]]), 
# array([[2518.869321]]), array([[38.88198276]]), array([[106.3686774]]), array([[24.60455618]]), 
# array([[24.31000214]]), array([[72.46302254]]), array([[118.20412401]]), array([[90.61706073]]),
#  array([[35.69280998]]), array([[32.29191778]])]

# [array([[295.92160086]]), array([[4586.07389789]]), array([[93918.70902541]]), array([[228.87564055]]), array([[71.45719435]]), array([[70.60730834]]), array([[37.76286348]]), array([[185.17609739]]), array([[6398.59278001]]), array([[45.69030508]]), array([[41.10979851]]), array([[3978.62048957]]), array([[76.10732595]]), array([[6559.51527665]]), array([[332.09798888]]), array([[49.90323791]]), array([[62.1688559]]), array([[16.95877941]]), array([[740.67677034]]), array([[861.35973576]]), array([[88.13572561]]), array([[14658.73651237]]), array([[8.38225161]]), array([[574.39402852]]), array([[1140.16647039]]), array([[382.91479485]]), array([[1042.69925719]]), array([[27.09653308]]), array([[61.25872228]]), array([[nan]])]   