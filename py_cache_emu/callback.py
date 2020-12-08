from .utils import log_utils


class Callback:
    def __init__(self, interval, **kwargs):
        self.interval = interval
        
        self.i_step = 0
        self.i_episode = 0
        self.kwargs = kwargs
    
    def reset(self):
        self.i_step = 0
        self.i_episode = 0
    
    def on_step_end(self, **kwargs):
        self.i_step += 1
        if self.i_step % self.interval == 0:
            self.i_episode += 1
            return self.on_episode_end(**kwargs)
    
    def on_episode_end(self, **kwargs):
        return {}
    
    def on_game_begin(self, **kwargs):
        return {}
    
    def on_game_end(self, **kwargs):
        return {}


class CallbackManager:
    def __init__(self, log_config={}, **kwargs):
        self.callbacks = []
        self.i_step = 0
        
        self.register_callback(LogCallback(**{**kwargs, **log_config}))
    
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def reset(self):
        for cb in self.callbacks:
            cb.reset()
    
    def on_step_end(self, **kwargs):
        self.i_step += 1
        
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_step_end(**kwargs)
            if res is not None:
                info.update(res)
        
        return info
    
    def on_game_begin(self, **kwargs):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_begin()
            if res is not None:
                info.update(res, **kwargs)
        
        return info
    
    def on_game_end(self, **kwargs):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_end(**kwargs)
            if res is not None:
                info.update(res)
        
        return info


class LogCallback(Callback):
    def __init__(self, interval=1000, main_tag="", sub_tag="", **kwargs):
        super(LogCallback, self).__init__(interval, **kwargs)
        self.main_tag = main_tag
        self.sub_tag = sub_tag
        self.verbose = kwargs.get("verbose", False)
        
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
        self.total_hit_cnt = 0
        self.total_req_cnt = 0
        
        self.episode_hit_cnts = []
        self.episode_req_cnts = []
        
        log_utils.init(**kwargs)
    
    def on_step_end(self, slice_hit_cnt=0, slice_req_cnt=0, **kwargs):
        self.episode_hit_cnt += slice_hit_cnt
        self.episode_req_cnt += slice_req_cnt
        self.total_hit_cnt += slice_hit_cnt
        self.total_req_cnt += slice_req_cnt
        
        super().on_step_end(**kwargs)
    
    def on_episode_end(self, **kwargs):
        self.episode_hit_cnts.append(self.episode_hit_cnt)
        self.episode_req_cnts.append(self.episode_req_cnt)
        
        episode_hit_rate = self.episode_hit_cnt / (self.episode_req_cnt + 1e-6)
        mean_hit_rate50 = sum(self.episode_hit_cnts[-50:]) / (sum(self.episode_req_cnts[-50:]) + 1e-6)
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        
        log_utils.write_scalars("0.MHR/{}".format(self.main_tag),
                                {self.sub_tag: mean_hit_rate},
                                self.i_episode, self.verbose)
        log_utils.write_scalars("1.MHR50/{}".format(self.main_tag),
                                {self.sub_tag: mean_hit_rate50},
                                self.i_episode, self.verbose)
        log_utils.write_scalars("2.EHR/{}".format(self.main_tag),
                                {self.sub_tag: episode_hit_rate},
                                self.i_episode, self.verbose)
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
    
    def on_game_end(self, **kwargs):
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        mean_hit_rate50 = sum(self.episode_hit_cnts[-50:]) / (sum(self.episode_req_cnts[-50:]) + 1e-6)
        return {
            "mean_hit_rate"  : "{:.1f}%".format(mean_hit_rate * 100),
            "mean_hit_rate50": "{:.1f}%".format(mean_hit_rate50 * 100),
            "total_hit_cnt"  : str(self.total_hit_cnt),
            "total_req_cnt"  : str(self.total_req_cnt)
        }
