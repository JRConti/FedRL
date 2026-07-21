# FedRL

# Problem: when comparing fed with centralized in main.py, we divide the training of 
# centralized in several blocks/rounds, but that is not equivalent to a centralized,
# as some internal params (eg exploration) depend on the unique call to learn() !

1) Lancer 1 DQN simple et vérifier qu'on a les mêmes perfs en monitoring que StableBaselines (https://wandb.ai/openrlbenchmark/sb3/workspace?nw=nwuseraraffin)

3) Clean (federated) code

4) Handle metrics monitoring : local -> server
4.1) In client.py: add the possibility to track agent's metrics 
Final check/objective: run Client.fit() and get metrics for one agent, ideally plot them on 
Weights&Biases as it is done in SB3 (plot in real training time)
4.2) In server.py: add the possibility to track some metrics for each agent 
Final check/objective: run Server.train() and get metrics for all agents, plot each metric (all agents superposed) on Weights&Biases (plot in real training time)

5) Aggregate individual agent's plots : mean, min, max --> mostly for the ep_mean_reward

6) Comparison for DQN: federated vs centralized vs federated w/o communication
Impact of the hyperparameters, change env (Frozen Lake, Cliff Walking, Atari ?) 

Later:

1) Other models than DQN

2) Other aggregation strategies

3) Parallel local trainings (torch.vfunc/vmap ?)
