from collections import Iterable

import numpy as np


class FeatureDict:
    def __init__(self):
        super().__init__()
        self.W = {}
    
    def clear(self):
        self.W.clear()
    
    def div_value(self, value):
        for key in self.W:
            self.W[key] /= value
        return self
    
    def get_value(self, key):
        return self.W[key]
    
    def set_value(self, key, value):
        self.W[key] = value
    
    def get_values(self, keys):
        values = np.zeros(len(keys), dtype=np.float)
        for i, key in enumerate(keys):
            if key in self.W:
                values[i] = self.W[key]
        return values
    
    def set_values(self, keys, values):
        if isinstance(values, Iterable):
            for key, value in zip(keys, values):
                self.W[key] = value
        else:
            for key in keys:
                self.W[key] = values
    
    def add_value(self, key, value):
        self.W[key] += value
    
    def add_values(self, keys, values):
        if isinstance(values, Iterable):
            for key, value in zip(keys, values):
                self.W[key] += value
        else:
            for key in keys:
                if key in self.W:
                    self.W[key] += values
                else:
                    self.W[key] = values


class NpFeatureDict(FeatureDict):
    def __init__(self, max_contents: int):
        super().__init__()
        self.max_contents = max_contents
        self.W = np.zeros(eval(max_contents), dtype=np.float)
    
    def div_value(self, value):
        self.W[:] /= value
        return self
    
    def get_values(self, keys):
        return self.W[keys]
    
    def set_values(self, keys, values):
        self.W[keys] = values
    
    def add_values(self, keys, values):
        self.W[keys] += values


class FeatureExtractor:
    def __init__(self, max_contents=-1, **kwargs):
        self.dim = 1
        if max_contents == -1:
            self.W = FeatureDict()
        else:
            self.W = NpFeatureDict(max_contents)
    
    def reset(self):
        self.W.clear()
    
    def forward(self, content_ids):
        features = self.W.get_values(content_ids)
        return features.reshape((len(content_ids), self.dim))
    
    def update(self, timestamps, content_ids):
        raise NotImplementedError()
