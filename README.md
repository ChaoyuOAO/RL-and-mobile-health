# RL-and-mobile-health

## Offline estimation--simulation
1. Simulate data for n patients with 2-d state vector over t time points along with the corresponding action lists and error matrix.
2. Compute policy probability matrix.
3. Get weights in $\Lambda_n$ from the policy probability of chosen actions ($\pi(A_i^t;S_i^t)$) and the known randomized trial ($\mu^t$).
4. Compute $\Lambda$ and get the $\hat{\theta}$ that minimize $\Lambda^T \Lambda+\lambda \theta^T \theta$ under a certain $\beta$

