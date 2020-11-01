import numpy as np

from .utils import NoneContentType


class Cache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        
        # 保存的内容，初始化为-1
        self._entries = np.zeros(capacity, dtype=np.int)
        self._entries[:] = NoneContentType
        
        # 保存内容存储的位置
        self._indices = {}
        
        # 保存内容访问的频率
        self._frequencies = {}
        
        # 保存空闲区域，用于空间分配
        self._free_indices = [idx for idx in range(self.capacity)]
    
    def reset(self):
        self._entries[:] = NoneContentType
        self._indices.clear()
        self._frequencies.clear()
        self._free_indices = [idx for idx in range(self.capacity)]
    
    def size(self):
        return len(self._indices)
    
    def full(self):
        return len(self._free_indices) == 0
    
    def empty(self):
        return len(self._free_indices) == self.capacity
    
    def hit_test(self, content_id: int):
        assert content_id != NoneContentType
        if content_id in self._frequencies:
            self._frequencies[content_id] += 1
        else:
            self._frequencies[content_id] = 1
        
        return content_id in self._indices
    
    def store(self, content_id: int):
        assert content_id != NoneContentType and content_id not in self._indices and not self.full()
        idx = self._free_indices.pop()
        self._entries[idx] = content_id
        self._indices[content_id] = idx
    
    def remove(self, content_id: int):
        assert content_id != NoneContentType and content_id in self._indices
        idx = self._indices[content_id]
        self._entries[idx] = NoneContentType
        del self._indices[content_id]
        self._free_indices.append(idx)
    
    def evict(self, idx: int):
        assert 0 <= idx < self.capacity
        self.remove(self._entries[idx])
    
    def get_contents(self):
        return self._entries
    
    def get_frequencies(self, content_ids=None):
        if content_ids is None:
            content_ids = self._entries
        
        frequencies = np.zeros_like(content_ids, dtype=np.float)
        for i, c_id in enumerate(content_ids):
            if c_id in self._frequencies:
                frequencies[i] = self._frequencies[c_id]
        
        self._frequencies.clear()
        
        return frequencies
