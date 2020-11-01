from collections import Iterable

import numpy as np


class FeatureDict:
    def __init__(self, **kwargs):
        self.W = {}
    
    def clear(self):
        self.W.clear()
    
    def div_value(self, value):
        for key in self.W:
            self.W[key] /= value
    
    def get_value(self, key):
        if key in self.W:
            return self.W[key]
        else:
            return 0
    
    def set_value(self, key, value):
        self.W[key] = value
    
    def add_value(self, key, value):
        if key in self.W:
            self.W[key] += value
        else:
            self.W[key] = value
    
    def get_values(self, keys):
        values = np.zeros(len(keys), dtype=np.float)
        for i, key in enumerate(keys):
            values[i] = self.get_value(key)
        return values
    
    def add_values(self, keys, values):
        if isinstance(values, Iterable):
            for key, value in zip(keys, values):
                self.add_value(key, value)
        else:
            for key in keys:
                self.add_value(key, values)


class NpFeatureDict(FeatureDict):
    def __init__(self, max_contents, **kwargs):
        super().__init__(**kwargs)
        self.max_contents = max_contents
        self.W = np.zeros(max_contents + 1, dtype=np.float)
    
    def clear(self):
        self.W[:] = 0
    
    def div_value(self, value):
        self.W[:] /= value
    
    def get_value(self, key):
        return self.W[key]
    
    def get_values(self, keys):
        return self.W[keys]
    
    def add_value(self, key, value):
        self.W[key] += value
    
    def add_values(self, keys, values):
        if isinstance(values, Iterable):
            for key, value in zip(keys, values):
                self.W[key] += value
        else:
            for key in keys:
                self.W[key] += values


class FeatureExtractor:
    def __init__(self, max_contents=-1, **kwargs):
        self.dim = 1
        if max_contents == -1:
            self.W = FeatureDict()
        else:
            self.W = NpFeatureDict(max_contents=max_contents)
    
    def reset(self):
        self.W.clear()
    
    def forward(self, content_ids):
        features = self.W.get_values(content_ids)
        return features.reshape((len(content_ids), self.dim))
    
    def update(self, timestamp, content_id):
        raise NotImplementedError()
    
    def update_batch(self, timestamps, content_ids):
        raise NotImplementedError()
