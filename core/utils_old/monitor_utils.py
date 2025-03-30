import os
import json
import copy
import psutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import threading
import time
from torch.utils.tensorboard import SummaryWriter


class ResourceMonitor:
    def __init__(self, log_dir, use_tensorboard=False, log_interval=60, save_interval=300):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        self._status = []
        self._start_time = None
        self._last_log_time = None
        self._last_save_time = None
        self._process = None

        self._last_status = {
            'disk': None,
            'network': None
        }

    def get_gpu_status(self):
        # Use nvidia-smi to get GPU status (Utility, Max Memory, Used Memory, Max Power, Used Power)
        result = subprocess.run(['nvidia-smi', '-i=0', '--query-gpu=utilization.gpu,memory.total,memory.used,power.limit,power.draw', '--format=csv,noheader,nounits'], capture_output=True)
        output = result.stdout.decode().strip().split(',')
        output = [float(x) for x in output]
        return {
            'utility(%)': output[0],
            'memory(GB)': {'total': output[1] / 1024, 'used': output[2] / 1024},
            'memory(%)': output[2] / output[1] * 100,
            'power(W)': {'limit': output[3], 'used': output[4]},
            'power(%)': output[4] / output[3] * 100
        }

    def get_cpu_status(self):
        # Use psutil to get CPU status (Utility)
        return {'utility(%)': psutil.cpu_percent()}

    def get_memory_status(self):
        # Use psutil to get memory status (Total Memory, Used Memory)
        memory = psutil.virtual_memory()
        return {
            'utility(GB)': {'total': memory.total / 1024**3, 'used': memory.used / 1024**3},
            'utility(%)': memory.percent
        }

    def get_network_status(self):
        # Use psutil to get network status (Total Network, Used Network)
        network = psutil.net_io_counters()
        if self._last_status['network'] is None:
            self._last_status['network'] = {'time': time.time(), 'bytes_sent': network.bytes_sent, 'bytes_recv': network.bytes_recv}
            return {'send(MB/s)': 0, 'recv(MB/s)': 0}
        else:
            new_status = {'time': time.time(), 'bytes_sent': network.bytes_sent, 'bytes_recv': network.bytes_recv}
            speed = {
                'send(MB/s)': (new_status['bytes_sent'] - self._last_status['network']['bytes_sent']) / (new_status['time'] - self._last_status['network']['time']) / 1024**2,
                'recv(MB/s)': (new_status['bytes_recv'] - self._last_status['network']['bytes_recv']) / (new_status['time'] - self._last_status['network']['time']) / 1024**2
            }
            self._last_status['network'] = new_status
            return speed

    def get_all_status(self):
        with ThreadPoolExecutor(max_workers=5) as executor:
            status = executor.map(
                lambda x: x(),
                [self.get_gpu_status, self.get_cpu_status, self.get_memory_status, self.get_network_status]
            )
        status = list(status)
        return {
            'gpu': status[0],
            'cpu': status[1],
            'memory': status[2],
            'network': status[3]
        }

    def _monitor(self):
        tb_writer = SummaryWriter(os.path.join(self.log_dir, 'resource_tb_logs')) if self.use_tensorboard else None
        while True:
            start_time = time.time()
            status = self.get_all_status()
            self._status.append((start_time, status))
            self._last_log_time = start_time
            if self.use_tensorboard:
                self._tb_log(status, tb_writer)
            if start_time - self._last_save_time > self.save_interval:
                threading.Thread(target=self._save_log, args=(copy.deepcopy(self._status),)).start()
                self._status = []
                self._last_save_time = start_time
            time.sleep(max(0, self._last_log_time + self.log_interval - time.time()))

    def _tb_log(self, status, tb_writer):
        _time = time.time() - self._start_time
        for resource_type in status.keys():
            for stat_name, stat_value in status[resource_type].items():
                if isinstance(stat_value, dict):
                    tb_writer.add_scalars(f'resource/{resource_type}/{stat_name}', stat_value, _time)
                else:
                    tb_writer.add_scalar(f'resource/{resource_type}/{stat_name}', stat_value, _time)

    def _save_log(self, status):
        with open(os.path.join(self.log_dir, 'resource_monitor.txt'), 'a') as f:
            for start_time, log in status:
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}: {json.dumps(log)}\n')

    def start(self):
        self._start_time = time.time()
        self._last_log_time = time.time()
        self._last_save_time = time.time()
        self._process = mp.Process(target=self._monitor)
        self._process.start()

    def stop(self):
        self._process.terminate()
        self._process.join()


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    monitor = ResourceMonitor('logs', use_tensorboard=True, log_interval=1, save_interval=5)
    monitor.start()
    time.sleep(12)
    monitor.stop()
            