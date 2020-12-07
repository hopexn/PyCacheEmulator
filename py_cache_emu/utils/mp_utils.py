import multiprocessing as mp

manager = mp.Manager()
_mutex = manager.Lock()

comm_size = 1
_buffers = manager.list([[]])
_barrier: mp.Barrier = mp.Barrier(comm_size)


def init(_comm_size):
    global comm_size
    global _buffers
    global _barrier
    comm_size = _comm_size
    _barrier = mp.Barrier(_comm_size)
    _buffers = manager.list([[] for _ in range(comm_size)])
    print("comm_size: {}".format(comm_size))

def all_to_all(data, rank):
    global _buffers
    global _barrier
    assert rank < comm_size
    
    _barrier.wait()
    _buffers[rank] = data
    _barrier.wait()
    return list(_buffers)
