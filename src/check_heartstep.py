import pandas as pd
import numpy as np
data = pd.read_csv("HS_MRT_example_v2.csv")
print(data.head())
## 13 19 34 not the same length.
action0 = data[["id", "MRT_action"]]
policy0 = data[["id", "MRT_probs"]]
utility0 = data[["id", "MRT_reward"]]
state0 = data[["id","dosage", "engagement","other.location","temperature"]]

def data_trans(df0,dim="single"):
    df0 = np.array(df0)
    n = np.unique(df0[:,0])
    if dim == "multi":
        df = {int(i): df0[df0[:,0]==i,1:] for i in n}
    elif dim == "single":
        df = {int(i): df0[df0[:,0]==i,1] for i in n}
    return df

action = data_trans(action0)
policy = data_trans(policy0)
utility = data_trans(utility0)
state = data_trans(state0)
for k in [13,19,24]:
    state.pop(k) 
id_new = range(37)
state_new = dict(zip(id_new,list(state.values()))) 
print({key: len(value) for key, value in state_new.items()})
print(len(state_new))
print(range(1,6))
