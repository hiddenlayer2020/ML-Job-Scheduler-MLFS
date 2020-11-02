def show_statistics(workload,cluster,timer):

  total_complete_time = 0
  num_deadline_satisfied = 0
  total_num_message = 0

  total_waiting_time = [] #0:size 100-300, 1: size 300-500, 2: size 500-700, 3: size 700-900
  num_job = [] #0:size 100-300, 1: size 300-500, 2: size 500-700, 3: size 700-900
  for i in range(4):
    total_waiting_time.append(0)
    num_job.append(0)
  
  acc_reduction = [] #0:0-0.15, 1:0.15-0.3, 2:0.3-0.45, 4:0.45-0.6, 5:0.6-0.75, 6:0.75-1 
  aver_jct_removed_ratio = [] #0:0-0.15, 1:0.15-0.3, 2:0.3-0.45, 3:0.45-0.6, 4:0.6-0.75, 5:0.75-1
  num_job_removed_ratio = [] #0:0-0.15, 1:0.15-0.3, 2:0.3-0.45, 3:0.45-0.6, 4:0.6-0.75, 5:0.75-1 

  for i in range(6):
    acc_reduction.append(0)
    aver_jct_removed_ratio.append(0)
    num_job_removed_ratio.append(0)


  #total_task = 0
  JCTs = []
  for job in workload:
    arriving_time = job.task_list[0].arriving_time
    complete_time = job.complete_time
    JCTs.append(complete_time - arriving_time)
  
    num_task = len(job.task_list)
    job_waiting_time = 0
  
    for task in job.task_list:
      #total_task += 1
      job_waiting_time += task.waiting_time
  
    if(job.input_size<325):#100-325
      total_waiting_time[0] += job_waiting_time / num_task
      num_job[0] += 1
    elif(job.input_size<550):#325-550
      total_waiting_time[1] += job_waiting_time / num_task
      num_job[1] += 1
    elif(job.input_size<775):#550-775
      total_waiting_time[2] += job_waiting_time / num_task
      num_job[2] += 1
    elif(job.input_size<1000):#775-1000
      total_waiting_time[3] += job_waiting_time / num_task
      num_job[3] += 1
  
    removed_ratio = job.num_removed_task / len(job.task_list)
    total_num_task = len(job.task_list)
    num_removed_task = job.num_removed_task

    if(removed_ratio < 0.15):
      acc_reduction[0] += 1 - (job.initial_loss - 0.372) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[0] += complete_time - arriving_time
      num_job_removed_ratio[0] += 1
    elif(removed_ratio < 0.3):
      acc_reduction[1] += 1 - (job.initial_loss - 0.372) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[1] += complete_time - arriving_time
      num_job_removed_ratio[1] += 1
    elif(removed_ratio < 0.45):
      acc_reduction[2] += 1 - (job.initial_loss - 0.372) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[2] += complete_time - arriving_time
      num_job_removed_ratio[2] += 1
    elif(removed_ratio < 0.6):
      acc_reduction[3] += 1 - (job.initial_loss - 0.372) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[3] += complete_time - arriving_time
      num_job_removed_ratio[3] += 1
    elif(removed_ratio < 0.75):
      acc_reduction[4] += 1 - (job.initial_loss - 0.372) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[4] += complete_time - arriving_time
      num_job_removed_ratio[4] += 1
    elif(removed_ratio < 1):
      acc_reduction[5] += 1 - (job.initial_loss - 0.35246) / (job.initial_loss - job.loss_traj[job.mloss_ind])
      aver_jct_removed_ratio[5] += complete_time - arriving_time
      num_job_removed_ratio[5] += 1
  
    jct = complete_time - arriving_time
    total_complete_time += jct
  
    if(jct < job.deadline):
      num_deadline_satisfied += 1
  
  sorted_JCTs = sorted(JCTs)
  num_job = len(workload)
  for i in range(num_job):
    jct = sorted_JCTs[i]
    perc = ((i+1)/num_job)*100
    print('CDF point( '+str(jct)+', '+str(perc)+'%)')

  avg_waiting_time = 0
  
  deadline_gurantee = num_deadline_satisfied / len(workload)
  print('job deadline gurantee = '+str(deadline_gurantee))

  print('remove ratio: 0-15%, 15-30%, 30-45%, 45-60%, 60-75%')

  aver_job_acc_red = 0
  
  for i in range(6):
    if(num_job_removed_ratio[i] == 0):
      acc_reduction[i] = 0
      aver_jct_removed_ratio[i] = 0
    else:
      acc_reduction[i] /= num_job_removed_ratio[i]
      aver_jct_removed_ratio[i] /= num_job_removed_ratio[i]
      aver_job_acc_red += acc_reduction[i]
      
    print('percentage of this part: '+str(num_job_removed_ratio[i] * 100 / len(workload))+'%')
    print('job accuracy reduction = '+str(acc_reduction[i] / len(workload)))
    print('average JCT over removed ratio = '+str(aver_jct_removed_ratio[i]))
  
  print('average job accuracy reduction: '+str(aver_job_acc_red / len(workload)))
  
  for i in range(4):
    if(num_job[i] != 0):
      avg_waiting_time += total_waiting_time[i]
      total_waiting_time[i] /= num_job[i]
    print('job waiting time = '+str(total_waiting_time[i]))
  print('average job waiting time = '+str(avg_waiting_time/len(workload)))
  average_jct = total_complete_time / len(workload)
  print('average jct = '+str(average_jct))

  total_num_message = 0

  for node in cluster.node_list:
    total_num_message += node.num_total_message
  
  avg_message = total_num_message/len(cluster.node_list)
  avg_bandwidth = avg_message * 0.0035
  print('avg_bandwidth = '+str(avg_bandwidth))
  
  total_time = 0
  
  for time in timer:
    total_time += time
  aver_time = total_time / len(timer)
  print('average latency = '+str(aver_time))

  
def is_schedule_all_task(job_list):
  for job in job_list:
    for task in job.task_list:
      if(not task.in_queue and not task.executing and not task.complete):
        return False
  return True
  
def schedule_policy(Gandiva, job_list, cluster):
  can_schedule = True
  if(not Gandiva):
    can_schedule = cluster.has_av_res()
  else:
    can_schedule = not is_schedule_all_task(job_list)
  return can_schedule