'''
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
'''
from sklearn.metrics import roc_auc_score
import cv2
import argparse
import sys
import os
import zipfile
import tensorflow as tf
import numpy as np
FLAGS = None

#HYPERPARAMETERS
# our photos are in the size of (80,80,3)
#Switching to CPU
if tf.test.gpu_device_name():
    print("GPU isn't gonna be used even if you have")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    print("No GPU Found")
    print("CPU is gonna be used")

#HYPERPARAMETERS
# our photos are in the size of (80,80,3)
IMG_SIZE = 80

epochs = 30
step_size = 8
IMG_SIZE_ALEXNET = 227
validating_size = 40
nodes_fc1 = 4096
nodes_fc2 = 4096
output_classes = 4

TRAIN_DIR = os.getcwd()

#Current working directory

print(TRAIN_DIR) # current working directory

#Unzipping file
with zipfile.ZipFile("datasets.zip","r") as zip_ref:
    zip_ref.extractall()

#Reading .npy files
train_data = np.load(os.path.join(os.getcwd(), 'datasets' ,'train_data_mc.npy'))
test_data = np.load(os.path.join(os.getcwd(), 'datasets' ,'test_data_mc.npy'))

#In order to implement ALEXNET, we are resizing them to (227,227,3)
for i in range(len(train_data)):
    train_data[i][0] = cv2.resize(train_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

for i in range(len(test_data)):
    test_data[i][0] = cv2.resize(test_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

train = train_data[:4800]
cv = train_data[4800:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
Y = np.array([i[1] for i in train])

cv_x = np.array([i[0] for i in cv]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
cv_y = np.array([i[1] for i in cv])
test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
test_y = np.array([i[1] for i in test_data])

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...



      #In order to implement ALEXNET, we are resizing them to (227,227,3)
      for i in range(len(train_data)):
        train_data[i][0] = cv2.resize(train_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

      for i in range(len(test_data)):
        test_data[i][0] = cv2.resize(test_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

      train = train_data[:4800]
      cv = train_data[4800:]



      X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
      Y = np.array([i[1] for i in train])

      cv_x = np.array([i[0] for i in cv]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
      cv_y = np.array([i[1] for i in cv])
      test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
      test_y = np.array([i[1] for i in test_data])

      steps = len(train)
      remaining = steps % step_size

      #Resetting graph
      tf.reset_default_graph()

      #Defining Placeholders
      x = tf.placeholder(tf.float32,shape=[None,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3])
      y_true = tf.placeholder(tf.float32,shape=[None,output_classes])

      ##CONVOLUTION LAYER 1
      #Weights for layer 1
      w_1 = tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.01))
      #Bias for layer 1
      b_1 = tf.Variable(tf.constant(0.0, shape=[[11,11,3,96][3]]))
      #Applying convolution
      c_1 = tf.nn.conv2d(x, w_1,strides=[1, 4, 4, 1], padding='VALID')
      #Adding bias
      c_1 = c_1 + b_1
      #Applying RELU
      c_1 = tf.nn.relu(c_1)

      print(c_1)
      ##POOLING LAYER1
      p_1 = tf.nn.max_pool(c_1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
      print(p_1)

      ##CONVOLUTION LAYER 2
      #Weights for layer 2
      w_2 = tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.01))
      #Bias for layer 2
      b_2 = tf.Variable(tf.constant(1.0, shape=[[5,5,96,256][3]]))
      #Applying convolution
      c_2 = tf.nn.conv2d(p_1, w_2,strides=[1, 1, 1, 1], padding='SAME')
      #Adding bias
      c_2 = c_2 + b_2
      #Applying RELU
      c_2 = tf.nn.relu(c_2)

      print(c_2)

      ##POOLING LAYER2
      p_2 = tf.nn.max_pool(c_2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
      print(p_2)

      ##CONVOLUTION LAYER 3
      #Weights for layer 3
      w_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
      #Bias for layer 3
      b_3 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 256, 384][3]]))
      #Applying convolution
      c_3 = tf.nn.conv2d(p_2, w_3,strides=[1, 1, 1, 1], padding='SAME')
      #Adding bias
      c_3 = c_3 + b_3
      #Applying RELU
      c_3 = tf.nn.relu(c_3)

      print(c_3)

      ##CONVOLUTION LAYER 4
      #Weights for layer 4
      w_4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01))
      #Bias for layer 4
      b_4 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 384][3]]))
      #Applying convolution
      c_4 = tf.nn.conv2d(c_3, w_4,strides=[1, 1, 1, 1], padding='SAME')
      #Adding bias
      c_4 = c_4 + b_4
      #Applying RELU
      c_4 = tf.nn.relu(c_4)

      print(c_4)

      ##CONVOLUTION LAYER 5
      #Weights for layer 5
      w_5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01))
      #Bias for layer 5
      b_5 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 256][3]]))
      #Applying convolution
      c_5 = tf.nn.conv2d(c_4, w_5,strides=[1, 1, 1, 1], padding='SAME')
      #Adding bias
      c_5 = c_5 + b_5
      #Applying RELU
      c_5 = tf.nn.relu(c_5)

      print(c_5)

      ##POOLING LAYER3
      p_3 = tf.nn.max_pool(c_5, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
      print(p_3)

      #Flattening
      flattened = tf.reshape(p_3,[-1,6*6*256])
      print(flattened)

      ##Fully Connected Layer 1
      #Getting input nodes in FC layer 1
      input_size = int( flattened.get_shape()[1] )
      #Weights for FC Layer 1
      w1_fc = tf.Variable(tf.truncated_normal([input_size, nodes_fc1], stddev=0.01))
      #Bias for FC Layer 1
      b1_fc = tf.Variable( tf.constant(1.0, shape=[nodes_fc1] ) )
      #Summing Matrix calculations and bias
      s_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
      #Applying RELU
      s_fc1 = tf.nn.relu(s_fc1)

      #Dropout Layer 1
      hold_prob1 = tf.placeholder(tf.float32)
      s_fc1 = tf.nn.dropout(s_fc1,keep_prob=hold_prob1)

      print(s_fc1)

      ##Fully Connected Layer 2
      #Weights for FC Layer 2
      w2_fc = tf.Variable(tf.truncated_normal([nodes_fc1, nodes_fc2], stddev=0.01))
      #Bias for FC Layer 2
      b2_fc = tf.Variable( tf.constant(1.0, shape=[nodes_fc2] ) )
      #Summing Matrix calculations and bias
      s_fc2 = tf.matmul(s_fc1, w2_fc) + b2_fc
      #Applying RELU
      s_fc2 = tf.nn.relu(s_fc2)
      print(s_fc2)

      #Dropout Layer 2
      hold_prob2 = tf.placeholder(tf.float32)
      s_fc2 = tf.nn.dropout(s_fc2,keep_prob=hold_prob1)

      ##Fully Connected Layer 3
      #Weights for FC Layer 3
      w3_fc = tf.Variable(tf.truncated_normal([nodes_fc2,output_classes], stddev=0.01))
      #Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
      b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
      #Summing Matrix calculations and bias
      y_pred = tf.matmul(s_fc2, w3_fc) + b3_fc
      #Applying RELU
      print(y_pred)

      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))
      matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
      acc = tf.reduce_mean(tf.cast(matches,tf.float32))

      global_step = tf.train.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          cross_entropy, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    #init = tf.global_variables_initializer()
    acc_list = []
    auc_list = []
    loss_list = []
    #saver = tf.train.Saver()

    config = tf.ConfigProto(device_count = {'GPU': 0})
    
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/home/ubuntu/TF-scheduler",
                                           hooks=hooks) as mon_sess:
      
      #Writing for loop to calculate test statistics. GTX 1050 isn't able to calculate all test data.
      '''
      cv_auc_list = []
      cv_acc_list = []
      cv_loss_list = []
      for v in range(0,len(cv_x)-int(len(cv_x) % validating_size),validating_size):
        acc_on_cv,loss_on_cv,preds = mon_sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)],
        feed_dict={x:cv_x[v:v+validating_size] ,y_true:cv_y[v:v+validating_size] ,hold_prob1:1.0,hold_prob2:1.0})
        print("global_step: "+str(global_step.eval(session = mon_sess)))
        auc_on_cv = roc_auc_score(cv_y[v:v+validating_size],preds)
        cv_acc_list.append(acc_on_cv)
        cv_auc_list.append(auc_on_cv)
        cv_loss_list.append(loss_on_cv)
        
      acc_cv_ = round(np.mean(cv_acc_list),5)
      auc_cv_ = round(np.mean(cv_auc_list),5)
      loss_cv_ = round(np.mean(cv_loss_list),5)
      acc_list.append(acc_cv_)
      auc_list.append(auc_cv_)
      loss_list.append(loss_cv_)
        
      print("Epoch:",i,"Accuracy:",acc_cv_,"Loss:",loss_cv_ ,"AUC:",auc_cv_)
      '''
      
      test_auc_list = []
      test_acc_list = []
      test_loss_list = []
      for v in range(0,len(test_x)-int(len(test_x) % validating_size),validating_size):
        mon_sess.graph._unsafe_unfinalize()
        acc_on_test,loss_on_test,preds = mon_sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)], 
      feed_dict={x:test_x[v:v+validating_size] ,
      y_true:test_y[v:v+validating_size] ,
      hold_prob1:1.0,
      hold_prob2:1.0})

        auc_on_test = roc_auc_score(test_y[v:v+validating_size],preds)
        test_acc_list.append(acc_on_test)
        test_auc_list.append(auc_on_test)
        test_loss_list.append(loss_on_test)

      test_acc_ = round(np.mean(test_acc_list),5)
      test_auc_ = round(np.mean(test_auc_list),5)
      test_loss_ = round(np.mean(test_loss_list),5)
      print("Test Results are below:")
      print("Accuracy:",test_acc_,"Loss:",test_loss_,"AUC:",test_auc_)
      
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
