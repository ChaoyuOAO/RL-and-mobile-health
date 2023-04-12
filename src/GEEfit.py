import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm, binom
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:/dingchaoyu_study/UM/dcy/M2health/SAVE/src/HS_MRT_example_v2.csv")
action0 = data[["id", "MRT_action"]]
policy0 = data[["id", "MRT_probs"]]
reward0 = data[["id", "MRT_reward"]]
state0 = data[["id","day","dosage","temperature","logpresteps","sqrt.totalsteps","variation","engagement"]]  #"engagement","other.location","variation","sqrt.totalsteps"
# ## scale dosage

full_data = data[["id","day","MRT_reward","dosage","engagement","temperature","logpresteps","other.location","variation","sqrt.totalsteps"]]
full_data.columns = ["id","day","MRT_reward","dosage","engagement","temperature","logpresteps","other_location","variation","sqrt_totalsteps"]
data_array = np.array(full_data)
# outcome = data_array[data_array[:,0]==1,1]
# plt.plot(range(len(outcome)),outcome)
# plt.show()
mod = sm.GEE.from_formula("MRT_reward ~ dosage+dosage+temperature+logpresteps+sqrt_totalsteps+engagement+other_location+variation", groups="id", data=full_data)
gee = mod.fit()
print(gee.summary())

# calculate the correlation matrix
corr = state0[["dosage","day","temperature","logpresteps","sqrt.totalsteps","variation", "engagement"]].corr()
print(corr)
# plot the heatmap
ax=sns.heatmap(corr, annot=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="crest")
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30)
plt.show()