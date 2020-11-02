import random
#!/usr/bin/env python
import psutil
from server import *
import threading
import socket
import server
from time import sleep
def get_cpu_memory():
  # gives a single float value
  cpu_usage = psutil.cpu_percent()
  #print('cpu percentage: '+str(cpu_usage)+'%')
  # gives an object with many fields
  psutil.virtual_memory()
  # you can convert that object to a dictionary 
  vm_dic = dict(psutil.virtual_memory()._asdict())
  memory_usage = vm_dic['percent']
  #print('memory percentage: '+str(memory_usage)+'%')
  
  return cpu_usage, memory_usage

class Cluster:
  node_list = []
  topology = {}
  completed_task = {} #job_name:[task_name, task_name, ...]
  task_distribution = {} #job_name:{node_name: num_task, node_name: num_task, ...}
  def __init__(self, num_rack, max_rack_size):
    self.num_rack = num_rack
    self.max_rack_size = max_rack_size
    
    for i in range(num_rack):
      self.topology[i] = []
  
  def process(self, cur_timestep):
    for node in self.node_list:
      pop_index = []
      if(len(node.queue) != 0):
        for i in range(len(node.queue)):
          task = node.queue[i]
          if(task.job_name not in node.job_list):
            node.add_task(task,cur_timestep)
            pop_index.append(i)
        for index in sorted(pop_index, reverse = True):
          node.queue.pop(index)

  def find_master(self):
    for node in self.node_list:
      if(node.master):
        return node  

  def add_node(self,node,rack):
    self.node_list.append(node)
    self.topology[rack].append(node)
        
  def has_av_res(self):
    for node in self.node_list:
      num_rtype = len(node.av_resource)
      node_available = True
      for i in range(num_rtype):
        if(node.av_resource[i] > 0.2):
          continue
        else:
          node_available = False
          break
      if(node_available):
        return True
      
    return False

  '''
  def update_process_rate(self):
    new_message_distribution = {} # node_name: num_new_message
    for i in range(len(self.node_list)):#initialization
      node_name = self.node_list[i].name
      new_message_distribution[node_name] = 0
      
    for job_name in self.task_distribution:
      task_map = self.task_distribution[job_name]
      num_task = 0
      for exe_node_name,num_node_task in task_map.items():
        num_task += num_node_task
      for exe_node_name,num_node_task in task_map.items():
        num_other_node_task = num_task - num_node_task
        num_new_message = num_node_task * num_other_node_task
        new_message_distribution[exe_node_name] += num_new_message
    
    for node in self.node_list:
      node_new_message = new_message_distribution[node.name]
      node.process_rate = 1 - 0.005 * node_new_message
      if(node.process_rate < 0.2):
        node.process_rate = 0.2
      node.num_total_message += node_new_message
  '''

  def step(self,time):
    #self.complete_task #job_name:[task_name, task_name, ... ]

    #for job_name, task_list in self.completed_task.items():
    #  self.task_distribution[job_name][node.name] -= len(task_list)

    return self.completed_task
          
  def complete_task(self,task,node):
    task_map = self.task_distribution[task.job_name]
    task_map[node.name] -= 1
    if(task_map[node.name] == 0):
      del task_map[node.name]
      if(len(task_map) == 0):
        del self.task_distribution[task.job_name]
  
  def find_task_on_node(self, worker_ip):
    for node in self.node_list:
      if(worker_ip == node.ip):
        return node

  def find_node(self, task):
    demand = task.demand
    min_affinity = len(demand) + 1
    selected_node = None
    available = True
    for node in self.node_list:
      cur_affinity = 0
      available = True
      for i in range(len(demand)):
        if(demand[i] < node.av_resource[i]):
          cur_affinity += node.av_resource[i] - demand[i]
        else:
          available = False
      if(available and cur_affinity < min_affinity):
        min_affinity = cur_affinity
        selected_node = node
    
    return selected_node
    
  def find_minload_node(self):
    num_rtype = len(self.node_list[0].av_resource)
    minload = 0
    first_node = self.node_list[0]
    selected_node = first_node
    for i in range(num_rtype):
      minload += 1 - first_node.av_resource[i]
      
    for node in self.node_list:
      cur_load = 0
      for i in range(num_rtype):
        cur_load += node.av_resource[i]
      if(cur_load < minload):
        minload = cur_load
        selected_node = node
    return selected_node


class Node:

  def __init__(self, name, num_rtype, ip):
    self.localIP = socket.gethostbyname(socket.gethostname())
    self.ip = ip
    self.master = (self.localIP == self.ip)
    self.job_list = []
    self.queue = []
    self.workload = None
    self.cluster = None
    self.scheduler = None
    

    self.name = name
    self.server_port = 9999
    self.port_availability = {}
    for port in range(2222,8888+1):
      self.port_availability[port] = True

    self.av_resource = []
    for i in range(num_rtype):
      self.av_resource.append(100)

    self.num_total_message = 0
    self.num_exe_task = 0
    self.num_total_task = 0
    self.task_list = []
    if(not self.master):
      start_server_cmd = '/home/ubuntu/TF-scheduler/remote_start_slave '+ip
      thread = threading.Thread(target = execute_command, args = (start_server_cmd,), name = 'Slave-Thread'+self.ip)
      thread.start()

  def start_master(self, workload, cluster, scheduler):
    
    localIP = socket.gethostbyname(socket.gethostname())
    Master(localIP, 9999, workload, cluster, scheduler).start()



  '''
  def process(self):
    completed_task = {} #job_name:[task_name, task_name, ... ]
    len_task_list = len(self.task_list)
    pop_index = []
    max_resource = 0
    num_rtype = len(self.av_resource)
    for i in range(num_rtype):
      if(max_resource < 1 - self.av_resource[i]):
        max_resource = 1 - self.av_resource[i]
        
    true_process_rate = self.process_rate
    
    if(max_resource > 1):
      true_process_rate *= 1 / max_resource
    
    for i in range(len_task_list):
      task = self.task_list[i]
      if(task.duration - true_process_rate <= 0):
        if(task.job_name not in completed_task):
          completed_task[task.job_name] = []
        self.num_exe_task -= 1
        task.duration = 0
        task.executing = False
        task.complete = True
        
        completed_task[task.job_name].append(task.task_name)
          
        num_rtype = len(self.av_resource)
        for j in range(num_rtype):
          self.av_resource[j] += task.demand[j]
          if(self.av_resource[j] > 1):
            self.av_resource[j] = 1
        pop_index.append(i)
        
          
      else:
        task.duration -= true_process_rate
    
    for ind in sorted(pop_index, reverse = True):
      self.task_list.pop(ind)
      
    return completed_task
  '''
  
  def get_available_port(self):
    for port in range(2222,8888+1):
      if(self.port_availability[port]):
        self.port_availability[port] = False
        return str(port)
    return -1

  def execute_task(self, task):

    worker_ip = task.worker_ip
    ps_port = task.ps_port
    ps_ip = task.ps_ip

    request = 'execute,'+ps_ip+':'+str(ps_port)+','+worker_ip+':'+str(task.occupied_worker_port)+','+task.job_name+','+str(task.index)

    response = TCP_request(ps_ip, 9999, request)
    print('execute task of '+task.job_name+' on worker: '+worker_ip+':'+str(task.occupied_worker_port))

  def add_task(self, task, cur_timestep):
    task.in_queue = False
    task.executing = True
    self.job_list.append(task.job_name)
    task.waiting_time = cur_timestep - task.arriving_time
    self.num_exe_task += 1
    self.num_total_task += 1
    #num_rtype = len(self.av_resource)
    #for i in range(num_rtype):
    #  self.av_resource[i] -= task.demand[i]
    self.task_list.append(task)
    self.execute_task(task)
  
  def is_started(self):
    request = 'is_started'
    response = 'not start'
    try:
      response = TCP_request(self.ip, 9999, request)
    finally:
      return (response == 'start')
  def clear_removed_task(self):
    if(len(self.queue) == 0):
      return
    pop_index = []
    for i in range(len(self.queue)):
      task = self.queue[i]
      if(task.complete):
        pop_index.append(i)

    for index in sorted(pop_index, reverse = True):
      self.queue.pop(index)


  def update_av_resource(self,cur_timestep):

    while(not self.is_started()):
      print(self.ip+' does not start, waiting...')
      sleep(5)

    request = 'cpu'
    response = TCP_request(self.ip, 9999, request)
    CPU = float(response)
    request = 'memory'
    response = TCP_request(self.ip, 9999, request)
    Memory = float(response)
    self.av_resource[0] = 100 - CPU
    self.av_resource[1] = 100 - Memory

    for task in self.queue:
      if(task.job_name not in self.job_list):
        self.add_task(task,cur_timestep)



def create_node(node_name, num_rtype, ip):
  node = Node(node_name, num_rtype, ip)
  return node

def create_cluster(max_rack, max_rack_size, max_num_nodes, num_rtype, ip_list):
  cur_num_node = 0
  cluster = Cluster(max_rack, max_rack_size)
  
  for i in range(max_num_nodes):
    cur_num_node += 1
    node_name = 'node'+str(cur_num_node)
    node = create_node(node_name,num_rtype, ip_list[i])
    rack = random.randint(0,max_rack-1)
    cluster.add_node(node,rack)
    
  return cluster
  
def display_available_port(port_availability):
  for port in range(2222,8888+1):
    if(not port_availability[port]):
      continue
    else:
      print('available ports: '+str(port)+'-8888')
      break

def display_node(node):
  print('node name: '+node.name)
  print('node ip: '+node.ip)
  display_available_port(node.port_availability)
  print('available resource percent: ')
  num_rtype = len(node.av_resource)
  for i in range(num_rtype):
    print(str(node.av_resource[i]))
  if(len(node.task_list) != 0):
    print('task list')
    for task in node.task_list:
      print('task '+str(task.task_name)+'\tjob '+task.job_name)
  else:
    print('task list empty')

  if(len(node.queue) != 0):
    print('node queue')
    for task in node.queue:
      print('task '+str(task.task_name)+'\tjob '+task.job_name)
  else:
    print('node queue empty')

  if(len(node.job_list) != 0):
    print('node job list')
    for job_name in node.job_list:
      print(job_name)
  else:
    print('node job list empty')
  print('\n')
      

def display_cluster(cluster, num_rtype):
  print('\nnumber of nodes: '+str(len(cluster.node_list)))
  for node in cluster.node_list:
    display_node(node)

def find_node(cluster, target_ip):
  for node in cluster.node_list:
    if(node.ip == target_ip):
      return node

if __name__ == "__main__":
  servers = {'node1':'172.31.6.117', 'node2':'172.31.4.135', 'node3':'172.31.3.225'}

  ip_list = []
  for ip in servers.values():
    ip_list.append(ip)

  max_num_rack = 3
  max_rack_size = 5
  max_num_nodes = len(servers)
  num_rtype = 2

  cluster = create_cluster(max_num_rack, max_rack_size, max_num_nodes, num_rtype, ip_list)
  display_cluster(cluster,2)