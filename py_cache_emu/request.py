import math
from math import ceil

import numpy as np

from .utils import proj_utils


class RequestSlice:
    def __init__(self, timestamps, content_ids, sparsity=1.0):
        assert len(timestamps) == len(content_ids)
        
        n_requests = len(timestamps)
        self.size = int(math.ceil(n_requests * sparsity))
        
        indices = sorted(np.random.choice(
            np.arange(n_requests),
            size=self.size,
            replace=(sparsity > 1.0)))
        
        self.timestamps = timestamps[indices]
        self.content_ids = content_ids[indices]
        
        self.ptr = 0
    
    def reset(self):
        self.ptr = 0
    
    def finished(self):
        return self.ptr >= self.size
    
    def next(self):
        assert not self.finished()
        timestamp, content_id = self.timestamps[self.ptr], self.content_ids[self.ptr]
        self.ptr += 1
        return timestamp, content_id


class RequestLoader:
    def __init__(self, data_path: str, time_beg, time_end, time_int=60, **kwargs):
        self.data = proj_utils.load_csv(data_path[kwargs.get("data_rank", 0)])
        
        self.n_requests = len(self.data)
        self.timestamps = self.data["timestamp"].to_numpy(dtype=np.int)
        self.content_ids = self.data["content_id"].to_numpy(dtype=np.int)
        
        self.sparsity = kwargs.get("sparsity", 1.0)
        self._slices = self._slice_by_time(self.timestamps, self.content_ids,
                                           time_beg, time_end, time_int,
                                           self.sparsity)
        self.i_slice = 0
        self.n_slices = len(self._slices)
        
        self.empty_slice = RequestSlice(np.zeros(0, dtype=np.int), np.zeros(0, dtype=np.int))
    
    def reset(self):
        self.i_slice = 0
        for s in self._slices:
            s.reset()
    
    def close(self):
        pass
    
    def _slice_by_time(self, timestamps, content_ids, t_beg, t_end, t_int, sparsity):
        t_beg, t_end, t_int = int(t_beg), int(t_end), int(t_int)
        assert t_end >= t_beg and t_int > 0 and len(timestamps) == len(content_ids)
        slices = []
        num_slices = int(ceil(float(t_end - t_beg) / t_int))
        ptr_beg, ptr_end = 0, 0
        last_time = t_beg
        for i in range(num_slices):
            next_time = last_time + t_int
            while ptr_end < self.n_requests and self.timestamps[ptr_end] < next_time:
                ptr_end += 1
            slices.append(RequestSlice(
                timestamps=timestamps[ptr_beg:ptr_end],
                content_ids=content_ids[ptr_beg:ptr_end],
                sparsity=sparsity
            ))
            ptr_beg = ptr_end
            last_time = next_time
        return slices
    
    def finished(self):
        return self.i_slice >= self.n_slices
    
    def next_slice(self):
        if self.finished():
            return self.empty_slice
        
        req_slice = self._slices[self.i_slice]
        self.i_slice += 1
        return req_slice
    
    def get_max_contents(self):
        return self.content_ids.max() + 1
