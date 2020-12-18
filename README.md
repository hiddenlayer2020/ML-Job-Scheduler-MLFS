# Job Scheduling for Large-Scale Machine Learning Clusters


MLFS is a machine learning job feature based job scheduler for machine learning clusters running both data parallelism and model parallelism machine learning jobs. The corresponding research work is published in CoNEXT 2020. Please click [here](https://dl.acm.org/doi/pdf/10.1145/3386367.3432588) for the detail of the paper.

### Prerequisites
- Install prerequisites (tested with Ubuntu 16.04, Tensorflow v1.8.0)
```
pip3 install -r (all the required softwares)
```

Required software: tensorflow==1.8.0, opencv-python==3.4.0.12, tqdm==4.19.6, pandas==0.22.0, matplotlib==2.2.0

numpy==1.16.2, scikit-learn==0.19.1, Python3==python 3.7.3

### Training
- To train a new model, put training data in `MLFS`, then in `sim/` run `python RLmodel.py` and then run
```
python train_RL_model.py
```

The reward signal and meta-setting of video can be modified in `RLmodel.py`. 

### Testing
- To test the trained model in simulated environment, first copy over the model to `test/models` and modify the `NN_MODEL` field of `test/train_RL_model.py` , and then in `test/` run `python evaluator.py` and then run 
```
python test.py
```

### Real-world experiments
- To run real-world experiments, distribute the whole folder to each physical machine within the cluster. Then, copy the trained RL model to `MLFS` and modify the `NN_MODEL` filed of `test/train_RL_model.py`. Next, update the IP addresses in `cluster.py` and `server.py`.  Finally, in `test` run
```
python real-test.py
```

The results will be saved to `test/results` folder.


### ACKNOWLEDGEMENTS
We thank the valuable suggestions from the anonymous reviewers and the shepherd. This research was supported in part by U.S. NSF grants, Microsoft Research Faculty Fellowship, and AWS Machine Learning Research Awards.



