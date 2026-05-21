# FedRL

1) Lancer DQN simple et vérifier qu'on a les mêmes perfs en monitoring que StableBaselines

2) Faire un DQN fédéré avec moyenne des poids comme agrégation (envs homogènes)
   - run 2 DQN trainings in sequential
   - run 2 DQN trainings in parallel (torch.vfunc ?)
   - do the aggregation for the 2 DQN each N iterations
