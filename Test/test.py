#!/usr/bin/env python

import random
import time

from statistics import *
from workload import *
from cluster import *
from scheduler import *
#from task import *
from RLmodel import *
from computation import *
import threading
import completion_task


max_res_demand = 0.3
min_res_demand = 0.1
min_in_size = 100
max_in_size = 1000
min_task_dur = 10
max_task_dur = 50
max_length_q = 100
max_num_nodes = 3
min_task = 10
max_task = 10

max_rack_size = 5
max_num_rack = 3
num_rtype = 2

max_q_length = 50
max_num_task = 200
min_num_task = 20
min_ddl = max_task_dur * min_task
max_ddl = max_task_dur * max_task
add_task_perc = 0.1
beta = 10 #percentage of tasks one time

W_q_d = 0.5
W_q_s = 0.5
W_q_r = 0.5

W_c_m = 0.5
W_c_t = 0.5

gama_q = 0.9
gama_n = 0.9
learning_rate = 0.1

remove_threshold = 0.1

num_coming_job = 5

remove_option = True
order_net_option = True
node_net_option = True

node_net_in_dim = 2 * max_num_nodes
node_net_out_dim = max_num_nodes

order_net_in_dim = 3 * max_q_length
order_net_out_dim = max_q_length

servers = {'node1':'172.31.6.117', 'node2':'172.31.4.135', 'node3':'172.31.3.225'}

ip_list = []
for ip in servers.values():
  ip_list.append(ip)

cluster = create_cluster(max_num_rack, max_rack_size, max_num_nodes, num_rtype, ip_list)

new_job = True

scheduler = Scheduler(max_q_length, add_task_perc, node_net_in_dim, node_net_out_dim, order_net_in_dim, order_net_out_dim)

num_episodes = 1
num_iteration = 1

wl_len = 10

Gandiva = False

ps_ip = socket.gethostbyname(socket.gethostname())
ps_node = find_node(cluster, ps_ip)
master = None
workload = generate_workload(wl_len, min_task, max_task, min_in_size, max_in_size, min_res_demand,max_res_demand,min_ddl,max_ddl,min_task_dur,max_task_dur,num_rtype, ps_node)
for node in cluster.node_list:
  if(node.master):
    master = node
    thread = threading.Thread(target = node.start_master, args = (workload, cluster, scheduler), name = 'Master-Thread'+node.ip)
    thread.start()
    break

timer = []
#workloads = []
#for i in range(num_episodes):
#  random.shuffle(workload)
#  workload_copy = copy_workload(workload)
#  workloads.append(workload_copy)

cluster.node_list[0].update_av_resource()

task = workload[0].task_list[0]
task.ps_port = master.get_available_port()
task.occupied_worker_port = 2222

for node in cluster.node_list:
  if(node.ip == '172.31.4.135'):
    task.worker_ip = node.ip
    node.add_task(task,0)
