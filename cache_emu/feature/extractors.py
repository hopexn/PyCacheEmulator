import math
import queue

import numpy as np

from .common import FeatureExtractor


class IdFeatureExtractor(FeatureExtractor):
    
    def forward(self, content_ids):
        return np.expand_dims(content_ids, axis=-1)
    
    def update(self, timestamp, content_id):
        pass
    
    def update_batch(self, timestamps, content_ids):
        pass


class RandomFeatureExtractor(FeatureExtractor):
    def forward(self, content_ids):
        return np.random.random((len(content_ids), self.dim))
    
    def update(self, timestamp, content_id):
        pass
    
    def update_batch(self, timestamps, content_ids):
        pass


class LfuFeatureExtractor(FeatureExtractor):
    def update(self, timestamp, content_id):
        self.W.add_value(content_id, 1)
    
    def update_batch(self, timestamps, content_ids):
        pass


class LruFeatureExtractor(LfuFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_time = 0
    
    def reset(self):
        super().reset()
        self.last_time = 0
    
    def update(self, timestamp, content_id):
        self.W.set_value(content_id, timestamp)
        self.last_time = timestamp
    
    def update_batch(self, timestamps, content_ids):
        pass


class OgdFeatureExtractor(LfuFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cnt = 0
    
    def reset(self):
        super().reset()
        self.cnt = 0
    
    def update(self, timestamp, content_id):
        eta = self.get_eta()
        self.W.add_value(content_id, eta)
        self.W.div_value(1 + eta)
        self.cnt += 1
    
    def update_batch(self, timestamps, content_ids):
        pass
    
    def get_eta(self) -> float:
        raise NotImplementedError


class OgdLruFeatureExtractor(OgdFeatureExtractor):
    def get_eta(self) -> float:
        return 1.0


class OgdLfuFeatureExtractor(OgdFeatureExtractor):
    def get_eta(self) -> float:
        return 1.0 / (self.cnt + 1)


class OgdOptFeatureExtractor(OgdFeatureExtractor):
    def get_eta(self) -> float:
        return 1.0 / math.sqrt(self.cnt + 1)


class SwfFeatureExtractor(FeatureExtractor):
    def __init__(self, history_w_len, **kwargs):
        super().__init__(**kwargs)
        self.history_w_len = history_w_len
        self.history = queue.Queue()
        self.n_history_requests = 0
    
    def reset(self):
        super().reset()
        self.history = queue.Queue()
        self.n_history_requests = 0
    
    def update(self, timestamp, content_id):
        self.W.add_value(content_id, 1)
        self.n_history_requests += 1
    
    def update_batch(self, timestamps, content_ids):
        self.history.put(content_ids)
        
        if self.history.qsize() > self.history_w_len:
            old_content_ids = self.history.get()
            self.W.add_values(old_content_ids, -1)
            self.n_history_requests -= old_content_ids.shape[0]
    
    def forward(self, content_ids):
        return super().forward(content_ids) / (self.n_history_requests + 1e-6)
