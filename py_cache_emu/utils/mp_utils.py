import multiprocessing as mp
import os

manager = mp.Manager()
_mutex = manager.Lock()

comm_size = 1
_buffers = manager.list([[]])
_barrier: mp.Barrier = mp.Barrier(comm_size)

_n_processes = manager.Value("n_processes", 0)
_process_map = manager.dict()


def init(_comm_size):
    global comm_size
    global _buffers
    global _barrier
    comm_size = _comm_size
    _barrier = mp.Barrier(_comm_size)
    _buffers = manager.list([[] for _ in range(comm_size)])


def register_process(rank=-1):
    global _buffers
    global _barrier
    global _n_processes
    global _process_map
    global comm_size
    
    pid = os.getpid()
    if pid not in _process_map:
        with _mutex:
            if pid not in _process_map:
                _process_map[pid] = rank if rank >= 0 else _n_processes.get()
                _n_processes.set(_n_processes.get() + 1)

def all_to_all(data):
    global _buffers
    global _barrier
    global _process_map
    
    pid = os.getpid()
    _barrier.wait()
    _buffers[_process_map[pid]] = data
    _barrier.wait()
    return list(_buffers)
