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
max_num_nodes = 4
min_task = 30
max_task = 50

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

remove_threshold = 0.0

num_coming_job = 5

remove_option = False
order_net_option = True
node_net_option = True

node_net_in_dim = 2 * max_num_nodes
node_net_out_dim = max_num_nodes

order_net_in_dim = 3 * max_q_length
order_net_out_dim = max_q_length

#servers = {'node1':'172.31.6.117', 'node2':'172.31.4.135', 'node3':'172.31.3.225'}
servers = {'node1':'172.31.6.117', 'node2':'172.31.44.243', 'node3':'172.31.35.92', 'node4':'172.31.35.219'}

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

workload = generate_workload(wl_len, min_task, max_task, min_in_size, max_in_size, min_res_demand,max_res_demand,min_ddl,max_ddl,min_task_dur,max_task_dur,num_rtype, ps_node)
for node in cluster.node_list:
  if(node.master):
    thread = threading.Thread(target = node.start_master, args = (workload, cluster, scheduler), name = 'Master-Thread'+node.ip)
    thread.start()
    break

timer = []
#workloads = []
#for i in range(num_episodes):
#  random.shuffle(workload)
#  workload_copy = copy_workload(workload)
#  workloads.append(workload_copy)

for iteration in range(num_iteration):
  print('iteration: '+str(iteration))
  #workload = generate_workload(wl_len,min_task,max_task,min_in_size,max_in_size,min_res_demand,max_res_demand,min_ddl,max_ddl,min_task_dur,max_task_dur,num_rtype)
  unfinished_job_list = workload
  queue_reward_traj = []
  node_reward_traj = []

  queue_sv_traj = []
  node_sv_traj = []

  queue_discounted_reward_traj = []
  node_discounted_reward_traj = []

  scheduler.node_network.zero_grad()
  scheduler.order_network.zero_grad()
  for episode in range(num_episodes):
    print('episode: '+str(episode))
    #workload = workloads[episode]
    queue_reward_traj.append([])
    node_reward_traj.append([])
    
    queue_sv_traj.append([])
    node_sv_traj.append([])
    
    queue_discounted_reward_traj.append([])
    node_discounted_reward_traj.append([])
    
    cur_timestep = time.clock()
    cur_job_num = 0
    num_schedule = 0
    schedule_interval = 5

    while(len(unfinished_job_list) != 0):
      
      cur_timestep = time.clock() + num_schedule * schedule_interval
      if(cur_job_num < wl_len):
        workload[cur_job_num].update_arr_time(cur_timestep)
        scheduler.job_arriving(workload[cur_job_num])
        cur_job_num += num_coming_job
        if(cur_job_num >= wl_len):
          cur_job_num = wl_len
      
      unfinished_job_list = find_unfinished_jobs(workload, cur_job_num, cur_timestep)

      if(len(unfinished_job_list) == 0):
        print('all jobs finish')
        break
      last_add_job = int(unfinished_job_list[0].job_name[3])
    
      valid = True
      num_decision = 0

      while(valid):
        
        dead_loop = 0
        
        can_schedule = schedule_policy(Gandiva, unfinished_job_list, cluster)

        while(can_schedule and len(scheduler.queue) < scheduler.max_q_length):

          dead_loop += 1
          
          last_add_job = scheduler.fill_queue(unfinished_job_list, last_add_job, dead_loop, remove_threshold, remove_option)
          if(last_add_job < 0):
            break

          can_schedule = schedule_policy(Gandiva, unfinished_job_list, cluster)
        
        start = time.clock()
        if(not Gandiva):
          cluster.process(cur_timestep)
          valid, q_sv, n_sv, t_ind, n_ind = scheduler.schedule_one_task(cur_timestep, cluster, order_net_option, node_net_option)
        else:
          valid = scheduler.Gandiva_schedule_one_task(cur_timestep, cluster)
        elapsed = (time.clock() - start)
        
        
        num_decision += 1
        
        if(len(scheduler.queue) == 0):
          #print('schedule all tasks')
          break
          
        if(not valid):
          #print('invalid action')
          num_decision -= 1
          break
          
        timer.append(elapsed)
        if(not Gandiva):
          queue_sv_traj[episode].append(q_sv)
          node_sv_traj[episode].append(n_sv)
          
          aver_size = scheduler.average_size()
          
          beta = scheduler.add_task_perc
          queue = scheduler.queue
          
          q_rt = queue_reward(W_q_d, W_q_s, W_q_r, cur_timestep, aver_size, scheduler, beta, queue, workload)
          queue_reward_traj[episode].append(q_rt)
          
          n_rt = node_reward(W_c_m, W_c_t, cur_timestep, cluster)
          node_reward_traj[episode].append(n_rt)

      print('\n\ncurrent time: '+str(cur_timestep))

      completion_task.shared_resource_lock.acquire()
      for job in workload:
        job.update_priority()
      for node in cluster.node_list:
        node.clear_removed_task()

      scheduler.clear_removed_task()
      display_scheduler(scheduler)
      display_cluster(cluster,2)
      for job in workload:
        display_job(job)

      print('scheduler now is sleeping... itertaion '+str(num_schedule))
      time.sleep(schedule_interval)

      completion_task.shared_resource_lock.release()
      num_schedule += 1
      
    print('finish episode '+str(episode)+', makespan = '+str(cur_timestep))    
    
    if(not Gandiva):
      num_action = len(queue_sv_traj[episode])
      for j in range(num_action):
        n_vt = discounted_reward(gama_n, j, node_reward_traj[episode])
        q_vt = discounted_reward(gama_q, j, queue_reward_traj[episode])
        node_discounted_reward_traj[episode].append(n_vt)
        queue_discounted_reward_traj[episode].append(q_vt)

    show_statistics(workload,cluster,timer)
    timer = []   
    
    unfinished_job_list = workload
  
  if(not Gandiva):
    num_action = 100000000
  
    for episode in range(num_episodes):
      if(num_action > len(queue_sv_traj[episode])):
        num_action = len(queue_sv_traj[episode])

    q_r_episode_baseline = []
    n_r_episode_baseline = []
    for j in range(num_action):
      total_qr_b = 0
      total_nr_b = 0
      for episode in range(num_episodes):
        total_qr_b += queue_discounted_reward_traj[episode][j]
        total_nr_b += node_discounted_reward_traj[episode][j]
      
      q_r_episode_baseline.append(total_qr_b / num_episodes)
      n_r_episode_baseline.append(total_nr_b / num_episodes)
  
    for episode in range(num_episodes):
      for j in range(num_action):
        q_sv = queue_sv_traj[episode][j]
        n_sv = node_sv_traj[episode][j]
        q_vt = queue_discounted_reward_traj[episode][j]
        n_vt = node_discounted_reward_traj[episode][j]
        qr_b = q_r_episode_baseline[j]
        nr_b = n_r_episode_baseline[j]
        scheduler.order_network.backward(q_sv, q_vt - qr_b, learning_rate)
        scheduler.node_network.backward(n_sv, n_vt - nr_b, learning_rate)
      
    scheduler.node_network.update_grads()
    scheduler.order_network.update_grads()
  
    torch.save(scheduler.node_network.weight, 'node_weight_ep'+str(iteration)+'.para')
    torch.save(scheduler.node_network.bias, 'node_bias_ep'+str(iteration)+'.para')
    torch.save(scheduler.order_network.weight, 'order_weight_ep'+str(iteration)+'.para')
    torch.save(scheduler.order_network.bias, 'order_bias_ep'+str(iteration)+'.para')