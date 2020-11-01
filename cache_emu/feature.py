import math
import queue

import numpy as np


class FeatureExtractor:
    def __init__(self, **kwargs):
        self.dim = 1
        self.W = {}
    
    def reset(self):
        self.W.clear()
    
    def forward(self, content_ids):
        features = np.zeros_like(content_ids, dtype=np.float)
        for i, c_id in enumerate(content_ids):
            if c_id in self.W:
                features[i] = self.W[c_id]
        return features.reshape((len(content_ids), self.dim))
    
    def update(self, timestamps, content_ids):
        raise NotImplementedError()


class IdFeatureExtractor(FeatureExtractor):
    def forward(self, content_ids):
        return np.expand_dims(content_ids, axis=-1)
    
    def update(self, timestamps, content_ids):
        pass


class RandomFeatureExtractor(FeatureExtractor):
    def forward(self, content_ids):
        return np.random.random((len(content_ids), self.dim))
    
    def update(self, timestamps, content_ids):
        pass


class LfuFeatureExtractor(FeatureExtractor):
    def update(self, timestamps, content_ids):
        if len(timestamps) == 0:
            return
        for t, c_id in zip(timestamps, content_ids):
            if c_id in self.W:
                self.W[c_id] += 1
            else:
                self.W[c_id] = 1


class LruFeatureExtractor(LfuFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_time = 0
    
    def reset(self):
        super().reset()
        self.last_time = 0
    
    def update(self, timestamps, content_ids):
        if len(timestamps) == 0:
            return
        for t, c_id in zip(timestamps, content_ids):
            self.W[c_id] = t
        self.last_time = timestamps[-1]


class OgdFeatureExtractor(LfuFeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cnt = 0
    
    def reset(self):
        super().reset()
        self.cnt = 0
    
    def update(self, timestamps, content_ids):
        if len(timestamps) == 0:
            return
        n_requests = len(content_ids)
        eta = self.get_eta()
        for t, c_id in zip(timestamps, content_ids):
            if c_id in self.W:
                self.W[c_id] += eta
            else:
                self.W[c_id] = eta
        self.cnt += 1
        
        for c_id in self.W:
            self.W[c_id] /= (1 + eta * n_requests)
    
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
    
    def update(self, timestamps, content_ids):
        self.history.put(content_ids)
        
        if len(timestamps) == 0:
            return
        
        for c_id in content_ids:
            if c_id in self.W:
                self.W[c_id] += 1
            else:
                self.W[c_id] = 1
        
        self.n_history_requests += len(content_ids)
        
        
        if self.history.qsize() > self.history_w_len:
            old_content_ids = self.history.get()
            for c_id in old_content_ids:
                if c_id in self.W:
                    self.W[c_id] -= 1
            self.n_history_requests -= len(old_content_ids)
    
    def forward(self, content_ids):
        return super().forward(content_ids) / (self.n_history_requests + 1e-6)


class FeatureManager:
    def __init__(self,
                 use_lru_feature=False, use_lfu_feature=False,
                 use_ogd_opt_feature=False, use_ogd_lru_feature=False, use_ogd_lfu_feature=False,
                 swf_w_lens=[]
                 ):
        self.extractors = []
        self.dim = 0
        
        if use_lru_feature:
            self.register_extractor(LruFeatureExtractor())
        if use_lfu_feature:
            self.register_extractor(LfuFeatureExtractor())
        if use_ogd_opt_feature:
            self.register_extractor(OgdOptFeatureExtractor())
        if use_ogd_lru_feature:
            self.register_extractor(OgdLruFeatureExtractor())
        if use_ogd_lfu_feature:
            self.register_extractor(OgdLfuFeatureExtractor())
        
        for swf_w_len in swf_w_lens:
            self.register_extractor(SwfFeatureExtractor(swf_w_len))
    
    def reset(self):
        for e in self.extractors:
            e.reset()
    
    def register_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)
        self.dim += extractor.dim
    
    def update(self, timestamps, content_ids):
        for e in self.extractors:
            e.update(timestamps, content_ids)
    
    def forward(self, content_ids):
        return np.concatenate([e.forward(content_ids) for e in self.extractors], axis=-1)
