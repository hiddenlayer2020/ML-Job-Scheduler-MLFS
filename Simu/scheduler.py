from RLmodel import *
from cluster import *
from workload import *
import logging
class Scheduler:

  def __init__(self,max_q_length,add_task_perc, node_net_in_dim, node_net_out_dim, order_net_in_dim, order_net_out_dim):
    self.order_net_in_dim = order_net_in_dim
    self.order_net_out_dim = order_net_out_dim
    self.node_net_in_dim = node_net_in_dim
    self.node_net_out_dim = node_net_out_dim
    self.node_network = RLnetwork(node_net_in_dim, node_net_out_dim)
    self.order_network = RLnetwork(order_net_in_dim, order_net_out_dim)
    
    self.queue = []
    self.max_q_length = max_q_length
    self.add_task_perc = add_task_perc
    
    
  def gen_q_state_vector(self,timestep): #(priortiy, input_size, arriving_time)
    q_sv = torch.Tensor(3 * self.max_q_length)
    for i in range(self.max_q_length):
      if(i<len(self.queue)):
        q_sv[2*i] = self.queue[i].priority
        q_sv[3*i+1] = self.queue[i].input_size
        q_sv[3*i+2] = timestep - self.queue[i].arriving_time
      else:
        q_sv[2*i] = 0
        q_sv[2*i+1] = 0
        q_sv[3*i+2] = 0
    return q_sv
  
  def sim(self, a, b):
    similarity = 0
    for i in range(len(a)):
      similarity += math.sqrt( (a[i] - b[i]) * (a[i] - b[i]) )
    return similarity
  
  def gen_n_state_vector(self, task, cluster):
    sv_dim = len(cluster.node_list) * 2
    n_sv = torch.Tensor(sv_dim)
    for i in range(len(cluster.node_list)):
        n_sv[i * 2] = 0
    if(task.job_name in cluster.task_distribution):
      task_map = cluster.task_distribution[task.job_name]
      for node_name, num_task in task_map.items():
        ind = (int(node_name[4]) - 1) * 2
        n_sv[ind] = num_task
            
    demand = task.demand
    for node in cluster.node_list:
      av_res = node.av_resource
      similarity = self.sim(av_res, demand)
      ind = (int(node.name[4]) - 1) * 2 + 1
      n_sv[ind] = similarity
    
    return n_sv
  
  def load_para(self, epoch, iteration):    
    self.node_network.weight = torch.load('node_weight_ep'+str(iteration)+'.para')
    self.node_network.bias = torch.load('node_bias_ep'+str(iteration)+'.para')
    self.order_network.weight = torch.load('order_weight_ep'+str(iteration)+'.para')
    self.order_network.bias = torch.load('order_bias_ep'+str(iteration)+'.para')
  
  def average_size(self):
    num_tasks = len(self.queue)
    total_size = 0
    for task in self.queue:
      total_size += task.input_size
    if(num_tasks == 0):
      return 0
    return total_size / num_tasks
  

  def job_arriving(self,job):
    
    cur_qlen = len(self.queue) # 0
    max_qlen = self.max_q_length # 5
    total_task_num = len(job.task_list) # 3
    add_task_num = math.ceil(total_task_num * self.add_task_perc) #int(1.2)

    if(add_task_num > max_qlen - cur_qlen):
      add_task_num = max_qlen - cur_qlen
      
    if(add_task_num > 0):
      cur_num_add_task = 0
      for task in job.task_list:
        if(not task.in_queue and not task.complete and not task.executing):
          self.queue.append(task)
          cur_num_add_task += 1
          task.in_queue = True
          if(cur_num_add_task >= add_task_num):
            break
    job.num_beta = 1

  def clear_removed_task(self):
    pop_index = []
    for i in range(len(self.queue)):
      task = self.queue[i]
      if(task.complete):
        pop_index.append(i)
    for index in sorted(pop_index, reverse = True):
      self.queue.pop(index)

  def get_num_scheduled_task(self,job):
    num_scheduled_task = 0
    for task in job.task_list:
      if(task.in_queue or task.complete or task.executing):
        num_scheduled_task += 1
    return num_scheduled_task
  
  def add_in_queue(self,job,threshold, dead_loop, remove_option = False):
    priority = 0
    for task in job.task_list:
      if(task.complete or task.executing):
        continue
      else:
        priority = task.priority
        break
    
    if(remove_option and priority < threshold):
      num_removed_task = 0
      for task in job.task_list:
        
        if(task.in_queue):
          num_removed_task += 1
          for j in range(len(self.queue)):
            q_task = self.queue[j]
            if(q_task.job_name == task.job_name and q_task.task_name == task.task_name):
              self.queue.pop(j)
              task.in_queue = False
              task.complete = True                
              break
        
        if(not task.complete and not task.executing):
          
          num_removed_task += 1
          task.in_queue = False
          task.complete = True

      return False
    
    job.num_beta += 1
    cur_qlen = len(self.queue) 
    max_qlen = self.max_q_length 

    total_num_task = job.num_task
    num_add_task = math.ceil(self.add_task_perc * total_num_task)
    num_scheduled_task = self.get_num_scheduled_task(job)
    num_executed_task = num_scheduled_task

    if(num_add_task > total_num_task - num_scheduled_task):      
      num_add_task = total_num_task - num_scheduled_task
      
    if(num_add_task > max_qlen - cur_qlen):      
      num_add_task = max_qlen - cur_qlen
      
    if(num_add_task == 0):  
      
      return False
    
    cur_num_add_task = 0 
    for task in job.task_list:
      
      if(task.in_queue or task.complete or task.executing):
        continue
      else:
        cur_num_add_task += 1
        self.queue.append(task)
        task.in_queue = True
        if(cur_num_add_task >= num_add_task):
          break
    if(cur_num_add_task == 0):
      return False
    
    if(cur_num_add_task < num_add_task):
      num_add_task = cur_num_add_task

    
    
    if(dead_loop > 10000):
      print('in add_in_queue')
      print('total_num_task = '+str(total_num_task))
      print('num_executed_task = '+str(num_executed_task))
      print('max_qlen = '+str(max_qlen))
      print('cur_qlen = '+str(cur_qlen))
      print('num_add_task = '+str(num_add_task))
      print('cur_num_add_task = '+str(cur_num_add_task))
      display_job(job)
      input()
    
    return True
  
  def Gandiva_schedule_one_task(self, timestep, cluster):
    q_ind = 0            
        
    if(len(self.queue) == 0):
      return False
    
    task = self.queue.pop(q_ind)

    node = cluster.find_node(task)
    
    if(node == None):
      node = cluster.find_minload_node()
    
    node.add_task(task, timestep)
        
    if(task.job_name in cluster.task_distribution):

      task_map = cluster.task_distribution[task.job_name]
      if(node.name in task_map):
        task_map[node.name] += 1
      else:
        task_map[node.name] = 1
        
    else:
      task_map = {}
      task_map[node.name] = 1
      cluster.task_distribution[task.job_name] = task_map
        
    return True
  
  def schedule_one_task(self, timestep, cluster, order_net_option = True, node_net_option = True):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format = LOG_FORMAT)
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    q_ind = 0
    
    q_sv = self.gen_q_state_vector(timestep)
    q_action = self.order_network.forward(q_sv)
    if(order_net_option):
      q_ind = torch.max(q_action,0)[1]
    
    else:
      q_ind = random.randint(0,len(self.queue))
         
    if(q_ind >= len(self.queue)):
      q_ind = len(self.queue) - 1
      if(q_ind < 0):
        logger.debug('return because q_ind < 0')
        return False, q_sv, None, q_ind, None
        
    task = self.queue.pop(q_ind)
    #task.in_queue = False
    n_sv = self.gen_n_state_vector(task,cluster)
    n_action = self.node_network.forward(n_sv)
    ind_list = []
    for i in range(len(cluster.node_list)):
      ind_list.append(i)
    
    ind = torch.Tensor(ind_list).int()
    
    if(node_net_option):
      _,ind = torch.topk(n_action,len(n_action),0,True,True)
    
    least_load_node = None
    least_tasks = 100000

    for node in cluster.node_list:
      num_rtype = len(node.av_resource)
      master = cluster.find_master()

      is_master = False
      if(node.ip == master.ip):
        is_master = True
      if(len(node.task_list)+len(node.queue) < least_tasks and len(node.queue) <= 5 and not is_master):
        least_load_node = node
        least_tasks = len(node.task_list)+len(node.queue)

    if(least_load_node == None):
      logger.debug('return because of none least load node')
      self.queue.append(task)
      return False, q_sv, n_sv, q_ind, -1
    logger.debug('least_load_node is '+str(least_load_node.name))
    for i in ind:
      n_ind = i.item()
      node = cluster.node_list[n_ind]

      if(node.ip != least_load_node.ip):
        node = least_load_node
      
      num_rtype = len(node.av_resource)
      is_available = True
      node.update_av_resource(timestep)
      for j in range(num_rtype):
        if(node.av_resource[j] < 0.2):
          is_available = False
      
      if(is_available):
        task.ps_port = master.get_available_port()
        task.worker_ip = node.ip
        task.occupied_worker_port = node.get_available_port()
        print('schedule task <'+task.job_name+':'+task.task_name+'> to '+node.name)
        if(task.job_name not in node.job_list):
          print('in schedule_one_task(), '+task.job_name+': '+task.task_name+' is not in node.job_list')
          print('node.job_list: '+str(node.job_list))
          node.add_task(task, timestep)
        else:
          node.queue.append(task)

        if(task.job_name in cluster.task_distribution):

          task_map = cluster.task_distribution[task.job_name]
          if(node.name in task_map):
            task_map[node.name] += 1
          else:
            task_map[node.name] = 1
        
        else:
          task_map = {}
          task_map[node.name] = 1
          cluster.task_distribution[task.job_name] = task_map

        return True, q_sv, n_sv, q_ind, n_ind
    self.queue.append(task)
    logger.debug('return because of none available node')
    return False, q_sv, n_sv, q_ind, n_ind

  
  def fill_queue(self, unfinished_job_list, last_add_job, dead_loop, remove_threshold, remove_option = False):
    num_job = len(unfinished_job_list)
    cur_add_job = (last_add_job + 1) % num_job
    
    if(dead_loop > 10000):
      print('dead loop in fill_queue()')
      print('num_job = '+str(num_job))
      print('cur_add_job = '+str(cur_add_job))
      print('queue length = '+str(len(self.queue)))
      input()
    
    if(len(self.queue) == self.max_q_length):
      return -1
    
    while(len(self.queue) < self.max_q_length):
      success = self.add_in_queue(unfinished_job_list[cur_add_job], remove_threshold, dead_loop, remove_option)
      if(not success):
        return -1
      
      cur_add_job = (cur_add_job + 1) % num_job
    
    return cur_add_job


def display_q(queue):
  print('queue info:')
  for i in range(len(queue)):
    display_task(queue[i])  


def display_scheduler(scheduler):
  print('\n\nscheduler info:')
  print('order_net_in_dim: '+str(scheduler.order_net_in_dim)+'\t order_net_out_dim: '+str(scheduler.order_net_out_dim)+
        '\t node_net_in_dim: '+str(scheduler.node_net_in_dim)+'\t node_net_out_dim: '+str(scheduler.node_net_out_dim)+
       '\n max_q_length: '+str(scheduler.max_q_length)+'\t add_task_perc: '+str(scheduler.add_task_perc)+
       '\t current_q_length: '+str(len(scheduler.queue)))

  display_q(scheduler.queue)