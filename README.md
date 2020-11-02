# Job Scheduling for Model-Parallelism Machine Learning Clusters

MLFS is a machine learning job feature based job scheduler for machine learning clusters running both data parallelism and model parallelism machine learning jobs.

### Prerequisites
- Install prerequisites (tested with Ubuntu 16.04, Tensorflow v1.8.0)
```
pip3 install -r (all the required softwares)
```

tensorflow==1.8.0

opencv-python==3.4.0.12

tqdm==4.19.6

pandas==0.22.0

matplotlib==2.2.0

numpy==1.16.2

scikit-learn==0.19.1

Python3==python 3.7.3



Steps to run the source code:

1. Install dependencies: pip3 install -r (all the required softwares)



2. Distribute this whole folder to each node involved in the implementation with the deployed environment above.

3. update the IP addresses in cluster.py and server.py. 

4. run the command "python3 train_RL_model"
