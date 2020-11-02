from sklearn.metrics import roc_auc_score
import cv2
import argparse
import sys
import os
import zipfile
import tensorflow as tf
import numpy as np
import time
from server import *
FLAGS = None
from completion_task import *

#HYPERPARAMETERS
# our photos are in the size of (80,80,3)
#Switching to CPU

def evaluate_model():
  pass;

def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    #pylint: disable=W0212
    session = session._sess
  return session

def mkdir(path):
  isExists = os.path.exists(path)
  if(not isExists):
    os.mkdir(path.decode('utf-8'))
    print('create directory success')
  else:
    print('directory exists')

def main(_):
  start = time.clock()

  #GPU settings
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.allocator_type = 'BFC'

  main_dir = '/home/ubuntu/TF-scheduler'

  job_dir = "/home/ubuntu/TF-scheduler/"+FLAGS.job
  cur_serial = 0

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  train_folder = FLAGS.train_folder
  task_index = FLAGS.task

  mkdir(main_dir+'/log_folder')
  mkdir(main_dir+'/loss_folder')
  mkdir(main_dir+'/accuracy_folder')

  j_name = FLAGS.job
  update_loss = 'NO'
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)


  if tf.test.gpu_device_name():
      print("GPU isn't gonna be used even if you have")
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
      print("No GPU Found")
      print("CPU is gonna be used")

  #HYPERPARAMETERS
  # our photos are in the size of (80,80,3)
  IMG_SIZE = 80

  IMG_SIZE_ALEXNET = 227
  validating_size = 40
  nodes_fc1 = 4096
  nodes_fc2 = 4096
  output_classes = 4

  TRAIN_DIR = os.getcwd()

  #Current working directory

  print(TRAIN_DIR) # current working directory

  ps_ip = ps_hosts[0].split(':')[0]
  request = 'checkpoint,'+job_dir
  response = TCP_request(ps_ip, 9999, request)
  if(response != 'none'):
    with open(job_dir+'/checkpoint', 'w') as f:
      f.write(response)

  #Unzipping file
  if(not os.path.exists(main_dir+"/datasets")):
    with zipfile.ZipFile(main_dir+"/datasets.zip","r") as zip_ref:
      zip_ref.extractall(main_dir)

  #Reading .npy files
  train_data = np.load(os.path.join(os.getcwd(), main_dir+'/datasets' ,'train_data_mc.npy'))
  #np.random.shuffle(train_data)
  test_data = np.load(os.path.join(os.getcwd(), main_dir+'/datasets' ,'test_data_mc.npy'))
  #np.random.shuffle(test_data)
  #In order to implement ALEXNET, we are resizing them to (227,227,3)
  for i in range(len(train_data)):
      train_data[i][0] = cv2.resize(train_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

  for i in range(len(test_data)):
      test_data[i][0] = cv2.resize(test_data[i][0],(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))

  train = train_data[:4800]
  cv = train_data[4800:]

  epochs = 1
  step_size = 8 #batch size
  num_batch = 600

  epoch_step = len(train)/step_size


  X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
  Y = np.array([i[1] for i in train])

  cv_x = np.array([i[0] for i in cv]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
  cv_y = np.array([i[1] for i in cv])

  test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
  test_y = np.array([i[1] for i in test_data])

  #test_x = np.array([i[0] for i in test_set]).reshape(-1,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3)
  #test_y = np.array([i[1] for i in test_set])

  if FLAGS.job_name == "ps":
    server.join()

  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...

      steps = len(train)
      remaining = steps % step_size

      #Resetting graph
      #tf.reset_default_graph()

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

      global_step = tf.train.get_or_create_global_step()

      #asynchronous training
      train_op = tf.train.AdamOptimizer(0.00001).minimize(
      cross_entropy, global_step=global_step)
      
      #synchronous training
      #opt = tf.train.AdagradOptimizer(0.00001)
      #opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=1, total_num_replicas=1)
      #train_op = opt.minimize(cross_entropy, global_step = global_step)

      matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
      acc = tf.reduce_mean(tf.cast(matches,tf.float32))

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    #init = tf.global_variables_initializer()
    acc_list = []
    auc_list = []
    loss_list = []
    saver = tf.train.Saver()

    #train_samples = 4800 images
    #1 mini-batch = 8 images
    #1 epoch = 4800 images / 8 (images/mini-batch) = 600 mini-batch = 1 task
    #600 step = 600 mini-batch = 1 task = 1 epoch
    Glo_step = (int(task_index) * num_batch) # 1 step = 1 mini-batch = 8 images

    config = tf.ConfigProto(device_count = {'GPU': 0})

    #num_epoch = 1

    #steps = 1000
    #os.mkdir(train_folder+'/'+j_name+'_checkpoint')
    #while num_epoch < epochs:

    

    try:
      start_training = time.clock()
      with tf.train.Session(master=server.target,
                                             is_chief=(FLAGS.task_index == 0),
                                             checkpoint_dir=train_folder,#+'/'+j_name+'_checkpoint',
                                             hooks=hooks) as mon_sess:

        #print('training, global_step: '+str(global_step.eval(session = mon_sess)))

        start_pos = Glo_step % epoch_step
        end_pos = start_pos + step_size * num_batch # one task is training batch_size * num_batch samples

        for j in range(start_pos,end_pos,step_size):
          #Feeding step_size-amount data with 0.5 keeping probabilities on DROPOUT LAYERS
          _, c = mon_sess.run([train_op, cross_entropy],
          feed_dict={x:X[j:j+step_size], y_true:Y[j:j+step_size],hold_prob1:0.5,hold_prob2:0.5})

          if(global_step.eval(session = mon_sess) % 20 == 0):
            with open('/home/ubuntu/TF-scheduler/log_folder/'+j_name+'_log', 'a') as f:
              f.write('\nstep: '+str(j)+"\tglobal_step: "+str(global_step.eval(session = mon_sess)))

        cur_serial = global_step.eval(session = mon_sess)

        Glo_step += num_batch
        print("global_step: "+str(Glo_step))
        epoch = int(Glo_step / epoch_step)
        if(Glo_step % epoch_step == 0):
          update_loss = 'YES'
          saver.save(get_session(mon_sess), train_folder+"/latest_model_"+j_name+"_epoch"+str(epoch)+".ckpt")
      
      end_training = time.clock()
      training_elapsed = end_training - start_training
      print('time elapsed on training'+str(training_elapsed))

      checkpoint_info = ''
      with open(job_dir+'/checkpoint', 'r') as f:
        checkpoint_info += f.read()

      request = 'update_checkpoint,'+checkpoint_info+','+job_dir
      response = TCP_request(ps_ip, 9999, request)
      print(response)

      #evaluate the model performance if trained a complete epoch
      if(Glo_step % epoch_step == 0):
        with tf.Session(target = server.target) as sess:

          with open('/home/ubuntu/TF-scheduler/log_folder/'+j_name+'_log', 'a') as f:
            f.write('\n\nevaluating the accuracy on the validation set, task index: '+str(task_index))
          saver.restore(sess, train_folder+"/latest_model_"+j_name+"_epoch"+str(epoch)+".ckpt")
          cv_auc_list = []
          cv_acc_list = []
          cv_loss_list = []
          for v in range(0,len(cv_x)-int(len(cv_x) % validating_size),validating_size):

            acc_on_cv, loss_on_cv, preds = sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)],
            feed_dict={x:cv_x[v:v+validating_size] ,y_true:cv_y[v:v+validating_size] ,hold_prob1:1.0,hold_prob2:1.0})

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
          with open('/home/ubuntu/TF-scheduler/log_folder/'+j_name+'_log', 'a') as f:     
            f.write("\nAccuracy:"+str(acc_cv_)+"\tLoss:"+str(loss_cv_)+"\tAUC:"+str(auc_cv_))
      
        #num_epoch += 1

        #Test the model performance after training
      
        with tf.Session(target = server.target) as sess:
      
          with open('/home/ubuntu/TF-scheduler/log_folder/'+j_name+'_log', 'a') as f:
            f.write('\n\ntest the model accuracy after training, task index: '+str(task_index))

          saver.restore(sess, train_folder+"/latest_model_"+j_name+"_epoch"+str(epoch)+".ckpt")
          test_auc_list = []
          test_acc_list = []
          test_loss_list = []
          for v in range(0,len(test_x)-int(len(test_x) % validating_size),validating_size):

            acc_on_test,loss_on_test,preds = sess.run([acc,cross_entropy,tf.nn.softmax(y_pred)], 
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
          with open('/home/ubuntu/TF-scheduler/log_folder/'+j_name+'_log', 'a') as f:
            f.write("\nTest Results are below:")
            f.write("\nAccuracy: "+str(test_acc_)+"\tLoss: "+str(test_loss_)+"\tAUC: "+str(test_auc_))

          with open('/home/ubuntu/TF-scheduler/loss_folder/loss_'+j_name, 'w') as f:
            f.write(str(test_loss_))
          with open('/home/ubuntu/TF-scheduler/accuracy_folder/accuracy_'+j_name, 'w') as f:
            f.write(str(test_acc_))

          with open('/home/ubuntu/TF-scheduler/loss_folder/loss_'+j_name+'_task'+task_index, 'w') as f:
            f.write(str(test_loss_))
          with open('/home/ubuntu/TF-scheduler/accuracy_folder/accuracy_'+j_name+'_task'+task_index, 'w') as f:
            f.write(str(test_acc_))

    except Exception as e:
      print('exception happens, this task fails')
      print(e)

    finally:
      role = FLAGS.job_name
      print('before send task completion')
      if(role == 'worker'):
        print('sending task completion')
        job_name = FLAGS.job
        task_index = FLAGS.task
        worker_host = FLAGS.worker_hosts
        worker_ip = worker_host.split(":")[0]
        worker_port = worker_host.split(":")[1]
        completion_task(job_name, task_index, worker_ip, worker_port, update_loss, job_dir, cur_serial)
      end = time.clock()
      exe_time = end - start
      print('whole execution time = '+str(exe_time))



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
  parser.add_argument(
      "--train_folder",
      type=str,
      default="/home/ubuntu/TF-scheduler",
      help="indicate the training directory"
  )  
  parser.add_argument(
      "--job",
      type=str,
      default="default",
      help="indicate the name of the job"
  )
  parser.add_argument(
      "--task",
      type=str,
      default="default",
      help="indicate the index of the task"
  )
  FLAGS, unparsed = parser.parse_known_args()
  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


  
  
