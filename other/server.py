
import socket               
import threading
import sys
import random
#!/usr/bin/env python
import psutil
import os
import completion_task
from workload import *
from cluster import *

def get_files(job_dir, file_keyword):
  files = []
  for file in os.listdir(job_dir):
    if(os.path.isfile(os.path.join(job_dir,file)) and file.startswith(file_keyword)):
      files.append(file)
  return files


def del_old_model(job_dir, cur_serial, task_index):
  files1 = get_files(job_dir, "model.ckpt")
  files2 = get_files(job_dir, "latest_model")
  for file in files1:
    partition = file.split('.')
    part2 = partition[1]
    serial_number = int(part2[5:])
    if(cur_serial - serial_number >=50):
      os.remove(job_dir+"/"+file)
  for file in files2:
    partition = file.split('.')
    part1 = partition[0]
    epoch = int(part1.split('_')[3][5:])
    if(epoch != task_index):
      os.remove(job_dir+"/"+file)

def getbytes(string):
  return string.encode()

def getstring(byte):
  return byte.decode()

def TCP_request(ip,port,request):
  c_s = socket.socket()
  response = 'request failed'
  try:
    c_s.connect((ip,int(port)))
    byte_request = getbytes(request)
    c_s.send(byte_request)
    byte_response = c_s.recv(1024)
    response = getstring(byte_response)

  except Exception, e:
    print('ip: '+ip+"\tport: "+str(port)+'\t request: '+request)
    print(e)

  finally:
    return response

def execute_command(cmd):
  os.system(cmd)


class Master:
  def __init__(self, ip, port, workload, cluster, scheduler):
    self.workload = workload
    self.sock = socket.socket()
    self.ip = ip
    self.port = port
    self.sock.bind((ip, int(port)))
    self.sock.listen(5)
    self.cluster = cluster
    self.scheduler = scheduler
    self.should_stop = False

  def TCP_reply(self, sock, addr):
    
    #print('new request from ' + str(addr) + ' is accepted')
    while (not self.should_stop):
      response = ''
      b_data=sock.recv(1024)
      data = getstring(b_data)
      if (data =='exit' or not data):
        break;

      elif(data =='ip'):
        response = str(self.ip)

      elif(data =='port'):
        response = str(self.port)

      elif(data == 'is_started'):
        response = 'start'
        
      elif(data == 'cpu'):
        cpu_usage = psutil.cpu_percent()
        response = str(cpu_usage)

      elif(data == 'memory'):
        vm_dic = dict(psutil.virtual_memory()._asdict())
        memory_usage = vm_dic['percent']
        response = str(memory_usage)

      elif(data.startswith('loss')):
        with open('/home/ubuntu/TF-scheduler/loss_folder/'+data,'r') as f:#data looks like: loss_jobname
          response = f.readline()

      elif(data.startswith('accuracy')):
        with open('/home/ubuntu/TF-scheduler/accuracy_folder/'+data,'r') as f:#data looks like: accuracy_jobname
          response = f.readline()
      elif(data.startswith('cluster')):
        response = 'nodes infomation'
        for node in self.cluster.node_list:
          response += 'node name: '+node.name+'\n'
          response += 'node ip: '+node.ip+'\n'

          if(len(node.task_list) != 0):
            response += 'task list\n'
            for task in node.task_list:
              response += 'task '+str(task.task_name)+'\tjob '+task.job_name+'\n'
          else:
            response += 'task list empty'

          if(len(node.queue) != 0):
            response += 'node queue\n'
            for task in node.queue:
              response += 'task '+str(task.task_name)+'\tjob '+task.job_name+'\n'
          else:
            response += 'node queue empty\n'


      elif(data.startswith('execute')):
        info = data.split(',')
        ps_addr = info[1]
        ps_ip = ps_addr.split(':')[0]
        ps_port = ps_addr.split(':')[1]
        worker_addr = info[2]
        worker_ip = worker_addr.split(':')[0]
        worker_port = worker_addr.split(':')[1]
        job_name = info[3]
        task_index = info[4]
        cmd = '/home/ubuntu/TF-scheduler/remote_task_executor '+ps_ip+' '+ps_port+' '+worker_ip+' '+worker_port+' '+job_name+' '+task_index
        
        thread = threading.Thread(target=execute_command, args=(cmd,), name = 'Task-Execution-Thread-'+worker_ip+'-'+job_name+'-'+task_index)
        thread.start()
        response = 'executing the task of '+job_name

      elif(data.startswith('checkpoint')):
        job_dir = data.split(',')[1]
        if(not os.path.exists(job_dir+'/checkpoint')):
          response = 'none'
        else:
          with open(job_dir+'/checkpoint', 'r') as f:
          	response = f.read()

      elif(data.startswith('update_checkpoint')):
      	partition = data.split(',')
      	checkpoint_info = partition[1]
      	job_dir = partition[2]
      	with open(job_dir+'/checkpoint', 'w') as f:
      	  f.write(checkpoint_info)
      	if(os.path.exists(job_dir+'/checkpoint')):
      	  response = 'update checkpoint file success!'
      	else:
      	  response = 'update failed'

      elif(data.startswith('task_completion')):
        info = data.split(',') #task_completion job_name task_index worker_ip worker_port
        job_name = info[1]
        job_index = int(job_name[3])
        task_index = int(info[2])
        worker_ip = info[3]
        worker_port = info[4]
        update_loss = info[5]
        job_dir = info[6]
        cur_serial = int(info[7])

        job = self.workload[job_index]
        task = job.task_list[task_index]

        if(update_loss == 'YES'):
          request = 'loss_'+job_name
          response = TCP_request(worker_ip, 9999, request)
          cur_loss = float(response)
          job.loss_traj.append(cur_loss)

        del_old_model(job_dir, cur_serial, task_index)

        task.executing = False
        task.complete = True

        completion_task.shared_resource_lock.acquire()
        if(not self.cluster.completed_task.has_key(job_name)):
          self.cluster.completed_task[job_name] = []

        self.cluster.completed_task[job_name].append(task.task_name)

        node = self.cluster.find_task_on_node(worker_ip)
        for i in range(len(node.task_list)):
          cur_task = node.task_list[i]
          if(cur_task.job_name == task.job_name and
            cur_task.task_name == task.task_name):
            node.task_list.pop(i)
            break
        for i in range(len(node.job_list)):
          cur_job_name = node.job_list[i]
          if(cur_job_name == task.job_name):
            node.job_list.pop(i)
            break
        #display_cluster(self.cluster, 2)
        completion_task.shared_resource_lock.release()
        print('port number is '+str(task.ps_port))
        response = 'processed task completion!'
        completion_task.stop_ps(task.ps_port)

      b_response = getbytes(response)
      sock.send(b_response)
    sock.close()
    #print('Connection from %s:%s closed' % addr)

  def start(self):
    while (True):
      c_sock, addr = self.sock.accept()
      thread = threading.Thread(target=self.TCP_reply, args=(c_sock,addr), name = 'Reuqest-Listening-Thread-'+str(addr))
      thread.start()

class Slave:
  def __init__(self, ip, port):
    self.sock = socket.socket()
    self.ip = ip
    self.port = str(port)
    self.sock.bind((ip, int(port)))
    self.sock.listen(5)
    self.should_stop = False


  def TCP_reply(self, sock, addr):
    
    #print('new request from ' + str(addr) + ' is accepted')
    while (not self.should_stop):
      response = ''
      b_data=sock.recv(1024)
      data = getstring(b_data)
      if (data =='exit' or not data):
        break;

      elif(data =='ip'):
        response = str(self.ip)

      elif(data =='port'):
        response = str(self.port)

      elif(data == 'is_started'):
        response = 'start'

      elif(data == 'cpu'):
        cpu_usage = psutil.cpu_percent()
        response = str(cpu_usage)

      elif(data == 'memory'):
        vm_dic = dict(psutil.virtual_memory()._asdict())
        memory_usage = vm_dic['percent']
        response = str(memory_usage)

      elif(data.startswith('loss')):
        with open('/home/ubuntu/TF-scheduler/loss_folder/'+data,'r') as f:#data looks like: loss_jobname
          response = f.readline()

      elif(data.startswith('accuracy')):
        with open('/home/ubuntu/TF-scheduler/accuracy_folder/'+data,'r') as f:#data looks like: accuracy_jobname
          response = f.readline()

      elif(data.startswith('execute')):
        info = data.split(',')
        ps_addr = info[1]
        ps_ip = ps_addr.split(':')[0]
        ps_port = ps_addr.split(':')[1]
        worker_addr = info[2]
        worker_ip = worker_addr.split(':')[0]
        worker_port = worker_addr.split(':')[1]
        job_name = info[3]
        task_index = info[4]
        cmd = '/home/ubuntu/TF-scheduler/remote_task_executor '+ps_ip+' '+ps_port+' '+worker_ip+' '+worker_port+' '+job_name+' '+task_index
        
        thread = threading.Thread(target=execute_command, args=(cmd,), name = 'Task-Execution-Thread-'+worker_ip+'-'+job_name+'-'+task_index)
        thread.start()
        response = 'executing the task of '+job_name

      b_response = getbytes(response)
      sock.send(b_response)
    sock.close()
    #print('Connection from %s:%s closed' % addr)

  def start(self):
    while (True):
      c_sock, addr = self.sock.accept()
      thread = threading.Thread(target=self.TCP_reply, args=(c_sock,addr), name = 'Reuqest-Listening-Thread-'+str(addr))
      thread.start()


if (__name__ == '__main__'):
  #print(len(sys.argv))
  if(len(sys.argv) != 2):
    print('usage: python server master/slave')
    print('exit')
    sys.exit(0)
  role = sys.argv[1]
  localIP = socket.gethostbyname(socket.gethostname())
  node = None

  if(role == 'master'):
    print('master')
    node = Master(localIP,9999)
    print('register')
    node.start()
    print('listening')

  elif(role == 'slave'):
    print('slave')
    node = Slave(localIP,9999)
    print('register')
    node.start()
    print('listening')

  else:
    print('request')
    response = TCP_request(localIP,9999,sys.argv[1])
    print("response is: "+response)


  
  
