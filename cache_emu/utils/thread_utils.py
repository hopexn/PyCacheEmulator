import threading

_mutex = threading.Lock()

comm_size = 1
_buffers = [[] for _ in range(comm_size)]
_barrier: threading.Barrier = threading.Barrier(comm_size)

_n_threads = 0
_thread_map = {}


def init(_comm_size):
    global comm_size
    global _buffers
    global _barrier
    comm_size = _comm_size
    _barrier = threading.Barrier(_comm_size)
    _buffers = [[] for _ in range(_comm_size)]


def register_thread():
    global _buffers
    global _barrier
    global _n_threads
    global _thread_map
    global comm_size
    tid = threading.get_ident()
    if tid not in _thread_map:
        with _mutex:
            if tid not in _thread_map:
                _thread_map[tid] = _n_threads
                _n_threads += 1
    assert _n_threads <= comm_size


def all_to_all(data):
    tid = threading.get_ident()
    _barrier.wait()
    _buffers[_thread_map[tid]] = data
    _barrier.wait()
    return _buffers.copy()
