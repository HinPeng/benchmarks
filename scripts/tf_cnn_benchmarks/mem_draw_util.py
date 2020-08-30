
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import collections
from deprecated import deprecated

decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)

# global vars
out_dir = './graph'
figure_dir = './figures'
mem_f = 'mem.json'

net_name = ''
bs = -1
stepid = -1
# _debug_log_num = 50

_ignore_devices = ['cpu', 'stream', 'memcpy']

# for matplotlib color
black = '#000000'

colors = ['tomato', 'darkcyan']


# overwrite option
# as tf.estimator.train() wraps all iterations' sess.run()
# the run_metadata is either enabled or disabled for all iterations
# for bert this overwrite is enabled
_overwrite = False
_except_nets = ['bert']

if not os.path.exists(out_dir):
  os.mkdir(out_dir)

if not os.path.exists(figure_dir):
  os.mkdir(figure_dir)

def _bad_devices(device_name):
  for i in _ignore_devices:
    if i in device_name:
      return True

  return False


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

class MemInfo(object):
  __slots__ = (
    'netname',
    'peak_mem',
    'temp_mem',
    'pers_mem',
    'mem_alloc_num',
    'temp_alloc_num',
    'pers_alloc_num',
  )

  def __init__(self, **kwargs):
    super(MemInfo, self).__init__()
    self.netname = None
    self.peak_mem = 0
    self.temp_mem = 0
    self.pers_mem = 0
    self.mem_alloc_num = 0
    self.temp_alloc_num = 0
    self.pers_alloc_num = 0

    for f, v in kwargs.items():
      setattr(self, f, v)

  def __repr__(self):
    content = ', '.join(['\"{}\" : \"{}\"'.format(f, getattr(self, f)) for f in MemInfo.__slots__])
    return '{'+content+'}'



def draw_alloc_infos(net,
                     batch_size,                     
                     run_metadata,
                     step_id=-1):
  assert run_metadata != None
  assert hasattr(run_metadata, 'step_stats')
  assert hasattr(run_metadata.step_stats, 'dev_stats')

  global net_name
  net_name = net
  global bs
  bs = batch_size
  global stepid
  stepid = step_id

  if net_name in _except_nets:
    _overwrite = True


  dev_stats = run_metadata.step_stats.dev_stats
  for dev_stat in dev_stats:
    device_name = _simplify_device_name(dev_stat.device)
    # pass the graph in cpu
    # devices name with 'stream' record the right execution time,
    # but the mem allocation is not in them
    if _bad_devices(device_name):
      # print(device_name)
      continue
    _get_alloc(device_name, dev_stat.node_stats, dumpfile=True)
      # pass

def _get_alloc(device_name, nodestats, dumpfile=False):
  allocs_info = []
  temp_mem = []
  pers_mem = []
  allocs_mem = []
  i = 0
  for node_stat in nodestats:
    node_name = node_stat.node_name
    if ':' in node_name:
      node_name = node_name.split(':')[0]

    mem_stat = node_stat.memory_stats
    tmp_mem = mem_stat.temp_memory_size
    persistent_mem = mem_stat.persistent_memory_size

    if tmp_mem != 0:
      temp_mem.append(float(tmp_mem)/(1<<20))
    if persistent_mem:
      pers_mem.append(float(persistent_mem)/(1<<20))

    # if debug:
    #     print("{}: temp mem: {}, persistent mem: {}".format(node_name, temp_mem, persistent_mem))

    alloc_memused = node_stat.memory
    # print("node_stat.memory size: {}".format(len(alloc_memused)))
    for alloc_mem in alloc_memused:
      alloc_name = alloc_mem.allocator_name
      if alloc_name.lower() != 'gpu_0_bfc':
        continue

      for alloc_rd in alloc_mem.allocation_records:
        alloc_micros = alloc_rd.alloc_micros
        alloc_bytes = alloc_rd.alloc_bytes
        allocs_info.append((alloc_micros, alloc_bytes))

  allocs_info.sort(key=lambda x: x[0])
  mem_alloc = []
  start = allocs_info[0][0]
  curr_mem = 0.0
  for _, data in enumerate(allocs_info):
    t = data[0] - start
    allocation_mem = float(data[1]) / (1<<20)  ## MB
    if allocation_mem > 0:
      allocs_mem.append(allocation_mem)
    curr_mem = curr_mem + allocation_mem / (1<<10)    ## GB
    mem_alloc.append((t, curr_mem))
    # t -= start
    # assert(t >= 0)
    # m = curr_mem + float(m) / (1<<30) # to gigabytes

  title = '{}_{}'.format(net_name, bs)
  x = [d[0] for d in mem_alloc]
  y = [d[1] for d in mem_alloc]
  alloc_num = 0
  for it in mem_alloc:
    if it[1] > 0:
      alloc_num += 1
  

  meminfo = MemInfo(netname=title,
                    peak_mem=max(y),
                    temp_mem=sum(temp_mem),
                    pers_mem=sum(pers_mem),
                    mem_alloc_num=alloc_num,
                    temp_alloc_num=len(temp_mem),
                    pers_alloc_num=len(pers_mem))
  maybewrite(mem_f, meminfo)
  _plot_alloc(title=title, x=x, y=y)
  _plot_alloc_cdf(title, allocs_mem)
  _plot_cdf(title, temp_mem, pers_mem)
  # labels = ['temporary memoy', 'persistent memory']
  # _plot_cdf(title, labels, p1=temp_mem, p2=pers_mem)

  if dumpfile:
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    # alloc_f = '{}/{}_{}_{}_alloc.log'.format(out_dir, net_name, bs, stepid)
    # temp_mem_f = '{}/{}_{}_{}_tmp_mem.log'.format(out_dir, net_name, bs, stepid)
    # pers_mem_f = '{}/{}_{}_{}_pers_mem.log'.format(out_dir, net_name, bs, stepid)
    alloc_f = '{}/{}_{}_alloc.log'.format(out_dir, net_name, bs)
    temp_mem_f = '{}/{}_{}_tmp_mem.log'.format(out_dir, net_name, bs)
    pers_mem_f = '{}/{}_{}_pers_mem.log'.format(out_dir, net_name, bs)
    if _overwrite or not os.path.exists(alloc_f):
      with open(alloc_f, 'w') as fout:
        for t, b in mem_alloc:
          fout.write('{}\t{}\n'.format(t, b))

    if _overwrite or not os.path.exists(temp_mem_f):
      with open(temp_mem_f, 'w') as fout:
        for m in temp_mem:
          fout.write('{}\n'.format(m))

    if _overwrite or not os.path.exists(pers_mem_f):
      with open(pers_mem_f, 'w') as fout:
        for m in pers_mem:
          fout.write('{}\n'.format(m))

def _plot_alloc(title, x, y):
  alloc_f = '{}/{}_alloc.pdf'.format(figure_dir, title)
  if not _overwrite and os.path.exists(alloc_f):
    return

  plt.scatter(x, y, s=5, c=black)
  plt.title(title)
  plt.ylabel('GPU Memory Usage (GB)')

  plt.savefig(alloc_f, format='pdf')
  plt.clf()

def _plot_alloc_cdf(title, data):
  alloc_cdf_f = '{}/{}_alloc_CDF.pdf'.format(figure_dir, title)
  if not _overwrite and os.path.exists(alloc_cdf_f):
    return

  data.sort()
  count = len(data)
  data_y = list(map(lambda x: float(x)/count, range(len(data))))
  plt.plot(data, data_y, color=black, linewidth=2.0, label='allocation memory')
  plt.xlabel('allocation memory')
  plt.title(title)
  plt.legend(loc='best')
  plt.savefig(alloc_cdf_f, format='pdf')
  plt.clf()


def _plot_cdf(title, a, b):
  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(figure_dir, title)
  if not _overwrite and os.path.exists(mem_cdf_f):
    return

  fig, ax1 = plt.subplots()

  a.sort()
  b.sort()
  count = len(a)
  data_y = list(map(lambda x: float(x)/count, range(len(a))))
  # plt.plot(a, data_y, color='tomato', linewidth=2.0, label='temp memory') 
  lg1, = ax1.plot(a, data_y, color='tomato', linewidth=2.0, label='temp memory')
  ax1.set_xlabel('temporary memory')
  ax1.set_ylabel(title)

  ax2 = ax1.twiny()
  count = len(b)
  data_y = list(map(lambda x: float(x)/count, range(len(b))))
  # plt.plot(b, data_y, color='darkcyan', linewidth=2.0, label='persistent memory')
  lg2, = ax2.plot(b, data_y, color='darkcyan', linewidth=2.0, label='persistent memory')
  ax2.set_xlabel('persistent memory')
  # plt.title(title)
  # plt.text
  # plt.xlabel('Memory size (MB)')
  # fig.suptitle(title, x=0, y=0.5, ha='left', va='center')
  
  # fig.legend(loc='center')
  plt.legend([lg1, lg2], ['temp memory', 'persistent memory'], loc='best')
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()
  # plt.legend(loc='best')
  # plt.savefig(mem_cdf_f, format='pdf')
  # plt.clf()

@deprecated
def _plot_cdf_error(title, labels, **kwargs):
  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(figure_dir, title)
  # if not _overwrite and os.path.exists(mem_cdf_f):
  #   return

  datas = kwargs.values()
  num = len(datas)

  ax = [0 for _ in range(num)]
  lg = [0 for _ in range(num)]
  fig, ax[0] = plt.subplots()

  for i, d in enumerate(datas):
    print(i, len(d))
    d.sort()
    count = len(d)
    if i > 0:
      # ax.append(ax[0].twiny())
      ax[i] = ax[0].twiny()
      
    data_y = list(map(lambda x: float(x)/count, range(len(d))))
    lg[i], = ax[i].plot(d, data_y, color=colors[i], linewidth=2.0, label=labels[i])
    ax[i].set_xlabel(labels[i])

    if i == 0:
      ax[i].set_ylabel(title)

  plt.legend(lg, labels, loc='best')  # can not place the legend in the 'best' location, 
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()
    

def _plot_alloc_f(title, filename):
  x = []
  y = []

  with open(filename) as fin:
    for line in fin:
      tmp = line.split('\t')
      x.append(int(tmp[0]))
      y.append(float(tmp[1]))

  plt.scatter(x, y, s=5, c=black)
  plt.title(title)
  plt.ylabel('GPU Memory Usage (GB)')

  alloc_f = '{}/{}_alloc.pdf'.format(figure_dir, title)
  plt.savefig(alloc_f, format='pdf')
  plt.clf()

def _plot_mem_f(title, temp_filename, pers_filename):
  temp_mem = []
  pers_mem = []
  with open(temp_filename) as fin:
    for line in fin:
      temp_mem.append(float(line))

  with open(pers_filename) as fin:
    for line in fin:
      pers_mem.append(float(line))

  temp_mem.sort()
  pers_mem.sort()
  
  fig, ax1 = plt.subplots()
  count = len(temp_mem)
  data_y = list(map(lambda x: float(x)/count, range(len(temp_mem))))
  lg1, = ax1.plot(temp_mem, data_y, color='tomato', linewidth=2.0, label='temp memory')
  ax1.set_xlabel('temporary memory')
  ax1.set_ylabel(title)
  # plt.legend(loc='best')

  ax2 = ax1.twiny()
  count = len(pers_mem)
  data_y = list(map(lambda x: float(x)/count, range(len(pers_mem))))
  lg2, = ax2.plot(pers_mem, data_y, color='darkcyan', linewidth=2.0, label='persistent memory')
  ax2.set_xlabel('persistent memory')

  mem_cdf_f = '{}/{}_mem_CDF.pdf'.format(figure_dir, title)
  # fig.legend(loc='best')
  plt.legend([lg1, lg2], ['temp memory', 'persistent memory'], loc='best')
  fig.savefig(mem_cdf_f, format='pdf')
  fig.clf()


def draw_file():
  configs = {
    'alexnet' : 512,
    'inception3' : 64,
    'inception4': 64,
    'resnet50' : 64,
    'resnet152': 64,
  }

  for c in configs.items():
    title = '{}_{}'.format(c[0], c[1])
    alloc_f = '{}/{}_alloc.log'.format(out_dir, title)
    temp_mem_f = '{}/{}_tmp_mem.log'.format(out_dir, title)
    pers_mem_f = '{}/{}_pers_mem.log'.format(out_dir, title)
    _plot_alloc_f(title, alloc_f)
    _plot_mem_f(title, temp_mem_f, pers_mem_f)
    


# def plot_file(filename):
#   x = []
#   y = []
#   with open(filename) as fin:
#     for line in fin:
#       temp = line.split('\t')
#       x.append(float(temp[0]))
#       y.append(float(temp[1]))

#   title = '_'.join(filename.split('/')[-1].split('_')[:2])
#   # print(title)
#   _plot_alloc(title, x=x, y=y)
#   maybewrite(mem_f, (title, max(y)))


# write to file if pair.key doesn't exist in file
def maybewrite(filename, meminfo):
  mem_info = []
  if os.path.exists('{}/{}'.format(out_dir, filename)):
    with open('{}/{}'.format(out_dir, filename)) as fin:
      mem_info = json.load(fin, object_pairs_hook=collections.OrderedDict)
  
  new = True
  for it in mem_info:
    if meminfo.netname == it['netname']:
      new = False
  
  if new:
    # print(eval(repr(meminfo)))    
    # mem_info.append(eval(repr(meminfo)))
    # print(decoder.decode(repr(meminfo)))
    mem_info.append(decoder.decode(repr(meminfo)))
    with open('{}/{}'.format(out_dir, filename), 'w') as fout:
      fout.write(json.dumps(mem_info, indent=2))


def test_plot():
  x = np.random.randn(20)
  y = np.random.randn(20)

  plt.scatter(x, y, s=5)
  plt.savefig('{}/{}.pdf'.format(figure_dir, 'test_scatter'))
  plt.clf()


if __name__ == "__main__":
  draw_file()