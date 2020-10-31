import sys
import threading
import server
#TCP_request(ip,port,request):

shared_resource_lock = threading.Lock()

def stop_ps(ps_host):
  cmd = "kill -9 $(ps -ef | grep " + str(ps_host) + " | grep 'job_name=ps' | awk '{print $2}')"
  server.execute_command(cmd)


def completion_task(job_name, task_index, worker_ip, worker_port, update_loss, job_dir, cur_serial):
  ip = '172.31.6.117'
  port = '9999'
  request = 'task_completion,'+job_name+','+task_index+','+worker_ip+','+worker_port+','+update_loss+','+job_dir+','+str(cur_serial)
  response = server.TCP_request(ip,port,request)
  print(response)
