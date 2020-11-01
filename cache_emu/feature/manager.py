from .extractors import *


class FeatureManager:
    def __init__(self,
                 use_lru_feature=False, use_lfu_feature=False,
                 use_ogd_opt_feature=False, use_ogd_lru_feature=False, use_ogd_lfu_feature=False,
                 use_id_feature=False, use_random_feature=False,
                 swf_w_lens=[], **kwargs
                 ):
        self.extractors = []
        
        if use_id_feature:
            self.register_extractor(IdFeatureExtractor(**kwargs))
        if use_lru_feature:
            self.register_extractor(LruFeatureExtractor(**kwargs))
        if use_lfu_feature:
            self.register_extractor(LfuFeatureExtractor(**kwargs))
        if use_ogd_opt_feature:
            self.register_extractor(OgdOptFeatureExtractor(**kwargs))
        if use_ogd_lru_feature:
            self.register_extractor(OgdLruFeatureExtractor(**kwargs))
        if use_ogd_lfu_feature:
            self.register_extractor(OgdLfuFeatureExtractor(**kwargs))
        if use_random_feature:
            self.register_extractor(RandomFeatureExtractor(**kwargs))
        
        for swf_w_len in swf_w_lens:
            self.register_extractor(SwfFeatureExtractor(swf_w_len, **kwargs))
    
    @property
    def dim(self):
        return len(self.extractors)
    
    def reset(self):
        for e in self.extractors:
            e.reset()
    
    def register_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)
    
    def update(self, timestamp, content_id):
        for e in self.extractors:
            e.update(timestamp, content_id)
    
    def update_batch(self, timestamps, content_ids):
        for e in self.extractors:
            e.update_batch(timestamps, content_ids)
    
    def forward(self, content_ids):
        return np.concatenate([e.forward(content_ids) for e in self.extractors], axis=-1)
