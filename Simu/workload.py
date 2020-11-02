import random
import math
import socket
from enum import Enum

class Job_Phase(Enum):
  Quick_drop = 0
  fluctuation = 1
  overfitting = 2

class Task:
  def __init__(self, task_name, job_name, deadline, input_size, duration, demand, ps_port, task_index):
    self.task_name = task_name
    self.job_name = job_name
    self.demand = demand
    self.deadline = deadline
    self.arriving_time = 0
    self.input_size = input_size
    self.priority = 1
    self.ps_ip = socket.gethostbyname(socket.gethostname())
    self.ps_port = ps_port
    self.index = task_index
    #self.duration = duration
    #self.pre_dur = duration
    self.worker_ip = None
    self.occupied_worker_port = None
    self.waiting_time = 0
    self.in_queue = False
    self.executing = False
    self.complete = False

class Job:

  def __init__(self, job_name, deadline, tasks, input_size, ps_port):
    self.mloss_ind = -1
    self.removed_task = []
    self.loss_phase = Job_Phase.Quick_drop
    self.pivot_index = 0
    self.samples = 0
    self.job_name = job_name
    self.task_list = tasks
    self.ps_ip = socket.gethostbyname(socket.gethostname())
    self.ps_port = ps_port
    self.input_size = input_size
    self.num_task = len(tasks)
    self.deadline = deadline
    self.complete_time = 0
    self.initial_loss = 1.41 # loss of model without training
    self.min_loss = self.initial_loss
    self.loss_traj = []
    self.num_beta = 0
    self.add_task_perc = 0.1
    self.num_removed_task = 0
    self.initialized = False
    #simulation loss trajectory
    #for i in range(self.num_task):      
    #  cur_loss = self.initial_loss * math.pow(0.85,i)
    #  self.loss_traj.append(cur_loss)
    
    self.arriving_time = 0
    self.complete_time = 0
  
  def cutoff_task(self):
    for task in self.task_list:
      if(task.in_queue):
        task.in_queue = False
        task.complete = True
        self.removed_task.append(task.task_name)
    self.num_removed_task = len(self.removed_task)

  def update_arr_time(self, timestep):
    if(self.arriving_time == 0):
      self.arriving_time = timestep
      for task in self.task_list:
        task.arriving_time = timestep
    
  def add_task(task):
    self.task_list.append(task)
    self.num_task += 1
    
  def count_complete_task(self):
    num_complete_task = 0
    for task in self.task_list:
      if(task.complete):
        num_complete_task += 1
        
    return num_complete_task
  
  def get_num_scheduled_task(self):
    num_scheduled_task = 0
    for task in self.task_list:
      if(task.in_queue or task.complete or task.executing):
        num_scheduled_task += 1
    return num_scheduled_task
  
  def detect_overfitting(self):
    L = len(self.loss_traj)
    local_optimals = self.find_local_optimals()
    loss_value_overfitting = False

    if(len(local_optimals) > 1):
      first_local_opt = list(local_optimals.values())[0]
      last_local_opt = list(local_optimals.values())[-1]
      if(last_local_opt > first_local_opt):
        loss_value_overfitting = True

    num_loin_epoch = 0 # loss increase epochs
    last_loss = self.loss_traj[-1]
    for i in range(-2, -L-1, -1):
      loss_change = last_loss - self.loss_traj[i]
      last_loss = self.loss_traj[i]
      if(loss_change > 0):
        num_loin_epoch += 1
      else:
        break 

      if(num_loin_epoch > 5 or loss_value_overfitting):
        self.loss_phase = Job_Phase.overfitting
        self.cutoff_task()

  def find_local_optimals(self):
    local_optimals = {} # {(index1:loss1),(index2:loss2),....}
    L = len(self.loss_traj)
    if(L < 3):
      return local_optimals

    for i in range(1,L-1):
      last_loss = self.loss_traj[i-1]
      cur_loss = self.loss_traj[i]
      next_loss = self.loss_traj[i+1]
      if(cur_loss < last_loss and cur_loss < next_loss):
        local_optimals[i] = cur_loss

    return local_optimals

  def update_priority(self):
    #num_complete_task = self.count_complete_task()
    num_complete_task = len(self.loss_traj)
    ind_last = num_complete_task - 1
    if(ind_last <= 0):#none finished task
      return

    num_total_task = self.num_task
    num_beta = self.num_beta
    
    num_scheduled_task = self.get_num_scheduled_task()
    ini_loss = self.initial_loss

    num_rest_task = num_total_task - num_complete_task
    last_loss_reduction = self.loss_traj[-2] - self.loss_traj[-1]     

    if(last_loss_reduction < 0 and self.loss_phase == Job_Phase.Quick_drop):
      self.loss_phase = Job_Phase.fluctuation
      self.pivot_index = num_complete_task - 2
      self.samples = 1
      self.min_loss = self.loss_traj[-2]
      self.mloss_ind = num_complete_task - 2

    if(self.loss_phase == Job_Phase.fluctuation):
      self.detect_overfitting()
      if(self.loss_phase == Job_Phase.overfitting):
        return
      self.samples = num_complete_task - self.pivot_index - 1
      num_fluc_task = num_total_task - self.pivot_index - 1
      max_num_samples = int(num_fluc_task * 0.37)
      if(self.samples > max_num_samples):
        if(self.min_loss > self.loss_traj[-1]):
          self.min_loss = self.loss_traj[-1]
          self.mloss_ind = num_complete_task - 1
          self.cutoff_task()
      elif(self.min_loss > self.loss_traj[-1]):
        self.min_loss = self.loss_traj[-1]
        self.mloss_ind = num_complete_task - 1
    

def display_task(task):
  print('task '+task.task_name)
  if(task.in_queue):
    print('status: in queue')
  elif(task.executing):
    print('status: executing')
  elif(task.complete):
    print('status: complete')
  else:
    print('status: not scheduled')


def create_demand(num_rtype,min_rd,max_rd):
  demand = []
  for i in range(num_rtype):
    demand.append(min_rd + (max_rd - min_rd) * random.random())
  return demand
  

def create_task(task_name, job_name, min_rd, max_rd, deadline, num_rtype,input_size, duration, flag_demand, demands, ps_port, task_index):
  demand = []
  if(flag_demand):
    demand = create_demand(num_rtype,min_rd,max_rd)
  else:
    demand = demands
  task = Task(task_name, job_name, deadline,input_size, duration, demand, ps_port, task_index)
  return task

def create_job(job_name, num_task, input_size, min_rd, max_rd, deadline, num_rtype, min_task_dur, max_task_dur, ps_node, flag_demand):
  tasks = []
  for i in range(num_task):
    task_name = str(i)
    demands = []
    duration = random.randint(min_task_dur,max_task_dur)
    ps_port = ps_node.get_available_port()
    task = create_task(task_name, job_name, min_rd, max_rd, deadline, num_rtype, input_size, duration, flag_demand, demands, ps_port, i)
    tasks.append(task)
  job = Job(job_name,deadline, tasks, input_size, ps_port)
  return job


def display_job(job):
  
  print('\n\njob name: '+job.job_name)
  print('validation loss:')
  if(len(job.loss_traj) == 0):
    print('no finished task')
  else:
    print(job.loss_traj)
  if(len(job.removed_task) == 0):
    print('no removed task')
  else:
    print('removed tasks: '+str(job.removed_task))
  for task in job.task_list:
    display_task(task)

import random
def generate_workload(length, min_task, max_task, min_in_size, max_in_size, min_rd, max_rd, min_ddl, max_ddl,min_task_dur,max_task_dur, num_rtype, ps_node):
  workload = []
  for i in range(length):
    job_name = 'job'+str(i)
    num_task = random.randint(min_task,max_task)
    input_size = random.randint(min_in_size,max_in_size)
    deadline = random.randint(min_ddl, max_ddl)

    job = create_job(job_name,num_task,input_size,min_rd,max_rd, deadline, num_rtype,min_task_dur,max_task_dur, ps_node, flag_demand = True)
    workload.append(job)
  return workload

def job_not_complete(job):
  for task in job.task_list:
    if(not task.complete):
      return True
  return False

def find_unfinished_jobs(workload, cur_job_num, timestep):
  unfinished_job_list = []
  for i in range(cur_job_num):
    if(job_not_complete(workload[i])):
      unfinished_job_list.append(workload[i])     
    elif(workload[i].complete_time < timestep and workload[i].complete_time == 0):
      workload[i].complete_time = timestep
  return unfinished_job_list


def copy_task(task):
  cp_task = Task(task.task_name, task.job_name, task.deadline, task.input_size, task.duration, task.demand, task.ps_port)
  return cp_task

def copy_job(job):
  cp_tasks = []
  for task in job.task_list:
    cp_task = copy_task(task)
    cp_tasks.append(cp_task)
  cp_job = Job(job.job_name, job.deadline, cp_tasks, job.input_size, job.ps_port)
  return cp_job

def copy_workload(workload):
  cp_wl = []
  for job in workload:
    cp_job = copy_job(job)
    cp_wl.append(cp_job)
  return cp_wl

