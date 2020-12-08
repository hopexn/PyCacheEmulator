import multiprocessing as mp

manager = mp.Manager()


class Comm:
    def __init__(self, comm_size):
        self.comm_size = comm_size
        self.barrier = mp.Barrier(comm_size)
        self.buffer = manager.list([[] for _ in range(comm_size)])
    
    def all_to_all(self, data, rank):
        self.barrier.wait()
        self.buffer[rank] = data
        self.barrier.wait()
        return list(self.buffer)


def init(comm_size):
    return Comm(comm_size)
