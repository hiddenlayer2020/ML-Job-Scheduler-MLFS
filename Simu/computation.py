import math
def average(a):
  total = 0
  num_element = len(a)
  for element in a:
    total += element
    
  return total / num_element

def std_dev(a):
  aver = average(a)
  total = 0
  for element in a:
    total += (element - aver) * (element - aver)
    
  return math.sqrt(total)

def get_num_message_list(cluster):
  num_message_list = []
  for node in cluster.node_list:
    num_message_list.append(node.num_total_message)
  return num_message_list

def get_num_task_list(cluster):
  num_task_list = []
  for node in cluster.node_list:
    num_task_list.append(node.num_total_task)
  return num_task_list

def get_remaining_time_list(queue,timestep):
  remaining_time_list = []
  for task in queue:
    remaining_time_list.append(task.arriving_time + task.deadline - timestep)
  return remaining_time_list

def get_input_size_list(queue):
  input_size_list = []
  for task in queue:
    input_size_list.append(task.input_size)
  return input_size_list

def node_reward(W_c_m, W_c_t, t,cluster):
  '''
  node.num_total_message
  node.num_exe_task 
  node.num_total_task
  '''
  reward_t = 0
  num_task_list = get_num_task_list(cluster)
  num_message_list = get_num_message_list(cluster)
  
  aver_num_task = average(num_task_list)
  std_num_task = std_dev(num_task_list)
  
  aver_num_message = average(num_message_list)
  std_num_message = std_dev(num_message_list)
  
  for node in cluster.node_list:
    r1 = 0
    r2 = 0
    
    if(std_num_message != 0):
      r1 = (node.num_total_message - aver_num_message) / std_num_message
      
    if(std_num_task != 0):
      r2 = (aver_num_task - node.num_total_task) / std_num_task
    
    reward_t += W_c_m * r1 + W_c_t * r2
    
  return reward_t
  
  
  
def queue_reward(W_q_d, W_q_s, W_q_r, t, aver_S, scheduler, beta, queue, workload):
  
  reward_t = 0
  input_size_list = get_input_size_list(queue)
  remaining_time_list = get_remaining_time_list(queue,t)
  
  
  aver_remaining_time = average(remaining_time_list)
  std_remaining_time = std_dev(remaining_time_list)
  
  aver_input_size = average(input_size_list)
  std_input_size = std_dev(input_size_list)
  
  
  for task in queue:
    job = workload[int(task.job_name[3])]
    num_scheduled_task = scheduler.get_num_scheduled_task(job)
    remaining_time = task.arriving_time + task.deadline - t
    r1 = 0
    r2 =0
    r3 = 0
    
    if(std_remaining_time != 0):
      r1 = W_q_d * (remaining_time - aver_remaining_time) / std_remaining_time
      
    if(std_input_size != 0):    
      r2 = W_q_s * (task.input_size - aver_input_size / std_input_size)
      
    r3 = W_q_r * (100 - num_scheduled_task*beta)/beta
    
    reward_t += r1 + r2 + r3

  return reward_t


def discounted_reward(gama,t,reward_traj):
  acc_reward = 0
  for ti in range(t):
    acc_reward += pow(gama,ti) * reward_traj[ti]
  return acc_reward