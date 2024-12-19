# MixRec: Heterogeneous Graph Collaborative Filtering

# Environment
The implementation for MixRec is under the following development environment:
- python=3.8.4
- tensorflow=1.14
- numpy=1.22.3
- scipy=1.7.3


# Datasets
We utilize three datasets for evaluating MixRec: Beibei, Tmall and IJCAI. We adopt two representative metrics for evaluating the accuracy of top-N item recommendations: Hit Ratio (HR@N) and Normalized Discounted Cumulative Gain (NDCG@N). Following the leave-one-out evaluation strategy, all negative samples are applied to construct the test set with the users’ all positive interactions under the target behavior type.
| Datasets | # Users | # Items | # Interactions	| Interaction Density |
| :-----| ----: | :----: | :----: | :----: |
| Beibei | 21716 | 7977 | 282860 | 0.1633% |
| Tmall | 114503 | 66706 | 491870  | 0.0064% |
| IJCAI | 423423 | 874328 | 2926616 | 0.0008% |


# Usage
Please unzip the Tmall and IJCAI dataset first. Also you need to create History/ and Models/ directories. Switch the working directory to `MixRec/`. The command lines to train it on the three datasets are as below. The un-specified hyperparameters in the commands are set as default. 
- Beibei
```
python mixrec_bei.py --data beibei --reg 1 --batch 32 
```
- Tmall
```
python mixrec.py --data tmall --ssl_reg 1e-6 --reg 5e-5 --keepRate 0.4 --graphSampleN 20000 --testgraphSampleN 40000
```
- IJCAI
```
python mixrec.py --data ijcai --lr 1e-4 --graphSampleN 20000 --testgraphSampleN 40000
```


# Important Arguments
- `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-2, 1e-3, 1e-4, 1e-5}`.
- `ssl_reg` and `sslGlobal_reg`: They are the weights for weight-decay regularization for the node-level and graph-level contrastive objectives. We tune this hyperparameter from the set `{1e-4, 1e-5, 1e-6, 1e-7}`.
- `graphSampleN`: This hyperparameter denotes the number of subgraph nodes in the train period. Recommended values are `{10000, 15000, 20000, 25000, 30000}`.
- `testgraphSampleN`: This hyperparameter denotes the number of subgraph nodes in the testing. Recommended values are `{30000, 35000, 40000, 45000, 50000}`.
