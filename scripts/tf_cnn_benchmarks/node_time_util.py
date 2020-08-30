import os

import tensorflow as tf

from node import Node
from tensor import Tensor

# gpu_nodes = dict()
dev_num = -1
nodes = []
cpu_nodes = dict()

out_dir = './graph/'


def _simplify_device_name(device):
  """/job:localhost/replica:0/task:0/device:CPU:0 -> /cpu:0"""

  prefix1 = '/job:localhost/replica:0/task:0/device:'
  prefix2 = '/device:'
  # suffix = 'stream:'
  if device.startswith(prefix1):
    device = device[len(prefix1):]
  elif device.startswith(prefix2):
    device = device[len(prefix2):]

  # if device.endswith(suffix, 0, -2):
  #   device = 
  if '/' in device:
    device = '_'.join(device.split('/'))

  if ':' in device:
    device = '_'.join(device.split(':'))

  return device.lower()

def get_node_time(run_metadata):

  assert run_metadata != None
  assert hasattr(run_metadata, 'step_stats')
  assert hasattr(run_metadata.step_stats, 'dev_stats')
  
  # 0: gpu_id_stream_all, 1: gpu_id
  dev_indices = []
  dev_stats = run_metadata.step_stats.dev_stats
  i = 0
  global dev_num
  for dev_stat in dev_stats:
    device_name = _simplify_device_name(dev_stat.device)
    if 'cpu' in device_name:
      extractCPUNodeTime(dev_stat.node_stats)
      extractCPUTensor(dev_stat.node_stats)
    if 'gpu' in device_name:
      # hardcode here: less than 10 devices
      dev_id = int(device_name[4])
      if (dev_id+1) > dev_num:
        dev_num = dev_id + 1
      while len(dev_indices) < (dev_id+1):
        tf.logging.info("Add GPU: %d" % (dev_id+1))
        dev_indices.append([-1]*2)
      if len(device_name) == 5:
        tf.logging.info("[GPU%d] gpu, index: %d" % (dev_id, i))
        dev_indices[dev_id][1] = i
      elif 'stream_all' in device_name:
        tf.logging.info("[GPU%d] gpu_stream_all, index: %d" % (dev_id, i))
        dev_indices[dev_id][0] = i
      else:
        pass      
    i += 1

  # Init nodes for each device (include gpu_nodes & cpu_nodes)
  for j in range(dev_num*2):
    nodes.append(dict())

  for dev_id in range(len(dev_indices)):
    dev_index = dev_indices[dev_id]

    if dev_index[0] != -1:
      gpu_stream_dev_stat = dev_stats[dev_index[0]]
      extractNodeTime(dev_id, gpu_stream_dev_stat.node_stats, True)
    else:
      tf.logging.info("Device %d don't have gpu_stream_all" % dev_id)

    gpu_dev_stat = dev_stats[dev_index[1]]
    extractNodeTime(dev_id, gpu_dev_stat.node_stats, False)
    extractTensor(dev_id, gpu_dev_stat.node_stats)
    
  # for dev_stat in dev_stats:
    # device_name = _simplify_device_name(dev_stat.device)

    # Ignore the node in CPU
    # if 'cpu' in device_name:
    #   continue
    # if device_name == 'gpu_0':
    # time in 'gpu_0' contains the cpu time, not gpu computation time
    # 'gpu_0' contains the output info, but 'gpu_0_stream_all' don't
    # if device_name == 'gpu_0_stream_all':
      # don't contain '_SOURCE' node
      # extractNodeTime(device_name, dev_stat.node_stats)
    
    # if device_name == 'gpu_0':
      # assert len(nodes) > 0
      # extractNodeTime(device_name, dev_stat.node_stats)
      # extractTensor(device_name, dev_stat.node_stats)

  # gpu_nodes_name = set(gpu_nodes.keys())
  # cpu_nodes_name = set(cpu_nodes.keys())
  # intersect_node_name = list(gpu_nodes_name.intersection(cpu_nodes_name))

  # if len(intersect_node_name) != 0:
  #   print("Intersection not empty!")
  #   print(str(intersect_node_name))

  PrintCPUResult()
  PrintResult()

# def extractTensor(device_name, nodestats):
#   for node_stat in nodestats:
#     node_name = node_stat.node_name
#     if gpu_nodes.__contains__(node_name):
#       gpu_nodes[node_name].InitOutput(node_stat)
#     elif cpu_nodes.__contains__(node_name):
#       cpu_nodes[node_name].InitOutput(node_stat)
#     else:
#       print("Error tensor: %s" % node_name)

def extractTensor(dev_id, nodestats):
  gpu_nodes = nodes[dev_id*2]
  cpu_nodes = nodes[dev_id*2+1]
  # logging.debug("gpu_nodes size: %d, cpu_nodes size: %d" % (len(gpu_nodes), len(cpu_nodes)))
  tf.logging.info("gpu_nodes size: %d, cpu_nodes size: %d" % (len(gpu_nodes), len(cpu_nodes)))
  for node_stat in nodestats:
    node_name = node_stat.node_name
    if gpu_nodes.__contains__(node_name):
      gpu_nodes[node_name].InitOutput(node_stat)
    elif cpu_nodes.__contains__(node_name):
      cpu_nodes[node_name].InitOutput(node_stat)
    else:
      # logging.error("Error tensor: %s" % node_name)
      tf.logging.error("Error tensor: %s" % node_name)

def extractCPUTensor(nodestats):
  for node_stat in nodestats:
    node_name = node_stat.node_name
    cpu_nodes[node_name].InitOutput(node_stat)
    

def extractCPUNodeTime(nodestats):
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  
  for node_stat in nodestats:
    node_name = node_stat.node_name
    if ':' in node_name:
      node_name = node_name.split(':')[0]
    # logging.debug("[CPU] Add node: %s" % node_name)
    tf.logging.info("[CPU] Add node: %s" % node_name)
    cpu_nodes[node_name] = Node(node_name)
    cpu_nodes[node_name].AddTime(node_stat)

def extractNodeTime(dev_id, nodestats, gpu_flag):
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  gpu_nodes = nodes[dev_id*2]
  cpu_nodes = nodes[dev_id*2+1]
  # logging.debug("gpu_nodes size: %d, cpu_nodes size: %d" % (len(gpu_nodes), len(cpu_nodes)))
  tf.logging.info("gpu_nodes size: %d, cpu_nodes size: %d" % (len(gpu_nodes), len(cpu_nodes)))
  for node_stat in nodestats:
    node_name = node_stat.node_name
    if ':' in node_name:
      node_name = node_name.split(':')[0]
      if gpu_flag:
        if not gpu_nodes.__contains__(node_name):
          gpu_nodes[node_name] = Node(node_name)
          # logging.debug("[GPU%d] Add gpu node: %s" % (dev_id, node_name))
          tf.logging.info("[GPU%d] Add gpu node: %s" % (dev_id, node_name))
        gpu_nodes[node_name].AddTime(node_stat)
      else:
        if gpu_nodes.__contains__(node_name):
          continue
        if not cpu_nodes.__contains__(node_name):
          cpu_nodes[node_name] = Node(node_name)
          # logging.debug("[GPU%d] Add cpu node: %s" % (dev_id, node_name))
          tf.logging.info("[GPU%d] Add cpu node: %s" % (dev_id, node_name))
        cpu_nodes[node_name].AddTime(node_stat)

# def extractNodeTime(device_name, nodestats):
#   # 
#   # assert hasattr(nodestats, 'referenced_tensor')
  

#   if not os.path.exists(out_dir):
#     os.mkdir(out_dir)

#   gpu_flag = False
#   if device_name == 'gpu_0_stream_all':
#     gpu_flag = True

#   # fout1 = open("%s%s_meta.txt" % (out_dir, device_name), 'w')
  
#   for node_stat in nodestats:
#     node_name = node_stat.node_name
#     if ':' in node_name:
#       node_name = node_name.split(':')[0]
#     if gpu_flag:
#       if not gpu_nodes.__contains__(node_name):
#         gpu_nodes[node_name] = Node(node_name)
#       gpu_nodes[node_name].AddTime(node_stat)
#     else:
#       if gpu_nodes.__contains__(node_name):
#         continue
#       if not cpu_nodes.__contains__(node_name):
#         cpu_nodes[node_name] = Node(node_name)
#       cpu_nodes[node_name].AddTime(node_stat)
    # if device_name == 'gpu_0':
    #   if nodes.__contains__(node_name):
    #     all_start_micros = node_stat.all_start_micros
    #     # fout1.write(node_name+' '+str(all_start_micros)+'\n')
    #     continue
    # if not nodes.__contains__(node_name):
    #   # if device_name == 'gpu_0':
    #   #   print("gpu_0 node: "+node_name)
    #   # else:
    #   #   print("stream_all: "+node_name)
    #   nodes[node_name] = Node(node_name)
    # nodes[node_name].AddTime(node_stat)

  # fout1.close()
    

# def PrintResult():
#   gpu_all_start_time = [node.start_time for node in gpu_nodes.values()]
#   cpu_all_start_time = [node.start_time for node in cpu_nodes.values()]
#   print("Min GPU all start time: %d" % min(gpu_all_start_time))
#   print("Min CPU all start time: %d" % min(cpu_all_start_time))
#   all_start_time = gpu_all_start_time+cpu_all_start_time
#   min_start_time = min(all_start_time)

#   with open('%s%s_nodetime.txt' % (out_dir, 'gpu_0_stream_all'), 'w') as fout:
#     for node in gpu_nodes.values():
#       assert (node.start_time >= min_start_time)
#       node.start_rel_time = node.start_time - min_start_time
#       node.end_time = node.start_rel_time + node.exec_time
#       fout.write(node.node_name+' '+str(node.start_rel_time)+' '+str(node.end_time)+'\n')

#   with open('%s%s_nodetime.txt' % (out_dir, 'gpu_0'), 'w') as fout:
#     for node in cpu_nodes.values():
#       assert (node.start_time >= min_start_time)
#       node.start_rel_time = node.start_time - min_start_time
#       node.end_time = node.start_rel_time + node.exec_time
#       fout.write(node.node_name+' '+str(node.start_rel_time)+' '+str(node.end_time)+'\n')

#   with open("%s%s_outputs.txt" % (out_dir, 'gpu_0'), 'w') as fout:
#     for node in gpu_nodes.values():
#       output_num = len(node.outputs)
#       fout.write("SrcNode"+' '+node.node_name+' '+str(output_num)+'\n')
#       for output in node.outputs.values():
#         fout.write("Output"+' '+str(output.tid)+' '+
#                    str(output.requested_bytes)+' '+
#                    str(output.allocated_bytes)+' '+                   
#                    str(output.allocator_name)+' '+
#                    str(output.allocated_time)+'\n')

#     for node in cpu_nodes.values():
#       output_num = len(node.outputs)
#       fout.write("SrcNode"+' '+node.node_name+' '+str(output_num)+'\n')
#       for output in node.outputs.values():
#         fout.write("Output"+' '+str(output.tid)+' '+
#                    str(output.requested_bytes)+' '+
#                    str(output.allocated_bytes)+' '+                   
#                    str(output.allocator_name)+' '+
#                    str(output.allocated_time)+'\n')

  # minimum_start_time = nodestats[0].all_start_micros
  # for node in nodestats:
  #   node_name = node.node_name
  #   all_start_micros = node.all_start_micros
  #   all_end_rel_micros = node.all_end_rel_micros

  #   if all_start_micros < minimum_start_time:
  #     minimum_start_time = all_start_micros

  #   d_node = Node(node_name, all_start_micros, all_end_rel_micros)

    # for ref_tensor in node.referenced_tensor:
    #   tid = ref_tensor.allocation_id
    #   allocator_name = ref_tensor.allocator_name
    #   requested_bytes = ref_tensor.requested_bytes
    #   allocated_bytes = ref_tensor.allocated_bytes

    #   t = Tensor(d_node.node_name, tid=tid, 
    #              requested_bytes=requested_bytes,
    #              allocator_name=allocator_name,
    #              allocated_bytes=allocated_bytes)
    #   d_node.ref_tensors.append(t)

    # for i in node.output:
    #   i_slot = i.slot
    #   td = i.tensor_description

    #   # dtype = td.dtype
    #   # dshape = td.shape
    #   allocation_d = td.allocation_description

    #   requested_bytes = allocation_d.requested_bytes
    #   allocated_bytes = allocation_d.allocated_bytes
    #   allocator_name = allocation_d.allocator_name

    #   t = Tensor(d_node.node_name, tid=i_slot,
    #              requested_bytes=requested_bytes,
    #              allocator_name=allocator_name,
    #              allocated_bytes=allocated_bytes,
    #              allocated_time=all_start_micros)

    #   d_node.outputs.append(t)

    # nodes.append(d_node)

def PrintCPUResult():
  with open("%scpu_outputs.txt" % out_dir, 'w') as fout:
    for node in cpu_nodes.values():
      output_num = len(node.outputs)
      fout.write("SrcNode"+' '+node.node_name+' '+str(output_num)+'\n')
      for output in node.outputs.values():
        fout.write("Output"+' '+str(output.tid)+' '+
                  str(output.requested_bytes)+' '+
                  str(output.allocated_bytes)+' '+                   
                  str(output.allocator_name)+' '+
                  str(output.allocated_time)+'\n')

def PrintResult():
  for dev_id in range(dev_num):
    gpu_nodes = nodes[dev_id*2]
    cpu_nodes = nodes[dev_id*2+1]
    gpu_all_start_time = [0]
    cpu_all_start_time = [0]
    if len(gpu_nodes) != 0:
      gpu_all_start_time = [node.start_time for node in gpu_nodes.values()]
    if len(cpu_nodes) != 0:
      cpu_all_start_time = [node.start_time for node in cpu_nodes.values()]
    print("Min GPU all start time: %d" % min(gpu_all_start_time))
    print("Min CPU all start time: %d" % min(cpu_all_start_time))
    all_start_time = gpu_all_start_time+cpu_all_start_time
    min_start_time = min(all_start_time)

    with open('%s%s_nodetime.txt' % (out_dir, ('gpu_%d_stream_all' % dev_id)), 'w') as fout:
      for node in gpu_nodes.values():
        assert (node.start_time >= min_start_time)
        node.start_rel_time = node.start_time - min_start_time
        node.end_time = node.start_rel_time + node.exec_time
        fout.write(node.node_name+' '+str(node.start_rel_time)+' '+str(node.end_time)+'\n')

    with open('%s%s_nodetime.txt' % (out_dir, ('gpu_%d' % dev_id)), 'w') as fout:
      for node in cpu_nodes.values():
        assert (node.start_time >= min_start_time)
        node.start_rel_time = node.start_time - min_start_time
        node.end_time = node.start_rel_time + node.exec_time
        fout.write(node.node_name+' '+str(node.start_rel_time)+' '+str(node.end_time)+'\n')

    with open("%s%s_outputs.txt" % (out_dir, ('gpu_%d' % dev_id)), 'w') as fout:
      for node in gpu_nodes.values():
        output_num = len(node.outputs)
        fout.write("SrcNode"+' '+node.node_name+' '+str(output_num)+'\n')
        for output in node.outputs.values():
          fout.write("Output"+' '+str(output.tid)+' '+
                    str(output.requested_bytes)+' '+
                    str(output.allocated_bytes)+' '+                   
                    str(output.allocator_name)+' '+
                    str(output.allocated_time)+'\n')

      for node in cpu_nodes.values():
        output_num = len(node.outputs)
        fout.write("SrcNode"+' '+node.node_name+' '+str(output_num)+'\n')
        for output in node.outputs.values():
          fout.write("Output"+' '+str(output.tid)+' '+
                    str(output.requested_bytes)+' '+
                    str(output.allocated_bytes)+' '+                   
                    str(output.allocator_name)+' '+
                    str(output.allocated_time)+'\n')
