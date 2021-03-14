
# author: muzhan
# contact: levio.pku@gmail.com
import os
import sys
import time
 
cmd = 'python ./cluster_baseline.py'

 
def gpu_info():
    # gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_status = os.popen('nvidia-smi').read().split('|')
    # get gpu memory info
    gpu1_info, gpu2_info = gpu_status[19], gpu_status[27]
    gpu1_info, gpu2_info = gpu1_info.strip(), gpu2_info.strip()
    gpu1_info, gpu2_info = gpu1_info.split(" / "), gpu2_info.split(" / ")
    # remove 'MiB'
    gpu1_use_mem, gpu1_total_mem = gpu1_info[0][:-3], gpu1_info[1][:-3]
    gpu2_use_mem, gpu2_total_mem = gpu2_info[0][:-3], gpu2_info[1][:-3]
    gpu1_remain_mem, gpu2_remain_mem = int(gpu1_total_mem) - int(gpu1_use_mem), int(gpu2_total_mem) - int(gpu2_use_mem)
    gpu_remain_mem = [gpu1_remain_mem, gpu2_remain_mem]
    return gpu_remain_mem
 
 
def narrow_setup(interval=2):
    gpu_remain_mem = gpu_info()
    i = 0
    while max(gpu_remain_mem) < 6000:  # set waiting condition
        gpu_remain_mem = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_memory_str = 'Remaining gpu memory: {} MiB |'.format(gpu_remain_mem)
        sys.stdout.write('\r' + gpu_memory_str + ' ' + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    gpu_index = gpu_remain_mem.index(max(gpu_remain_mem))
    sys.stdout.write("GPU Max Remain: {}".format(max(gpu_remain_mem)) + ' In GPU Number {}'.format(gpu_index))
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('\n' + cmd)
    os.system(cmd)
 
 
if __name__ == '__main__':
    narrow_setup()
