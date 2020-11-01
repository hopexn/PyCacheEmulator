import math

from .callback import Callback
from .tqdm_callback import TqdmCallback


class CallbackManager:
    def __init__(self, total_steps, n_episode_steps: int = 1000,
                 enable_tqdm=False, enable_log=False, **kwargs):
        self.callbacks = []
        self.total_steps = total_steps
        self.n_episode_steps = n_episode_steps
        self.n_episode = int(self.total_steps / self.n_episode_steps)
        self.step_count = 0
        
        if enable_tqdm:
            self.callbacks.append(TqdmCallback(self.n_episode))
        
        assert (self.n_episode_steps > 0)
    
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_step_end(self, **kwargs):
        self.step_count += 1
        
        info = {}
        
        for cb in self.callbacks:
            cb.on_step_end()
        
        if self.step_count % self.n_episode_steps == 0:
            for cb in self.callbacks:
                res = cb.on_episode_end(**kwargs)
                if res is not None:
                    info.update(res)
        
        return info
    
    def on_episode_end(self):
        pass
    
    def on_game_begin(self):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_begin()
            if res is not None:
                info.update(res)
        
        return info
    
    def on_game_end(self):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_game_end()
            if res is not None:
                info.update(res)
        
        return info
    
    def switch_mode(self, test_mode=False):
        for cb in self.callbacks:
            cb.switch_mode(test_mode)
        print("Mode: {}".format("test" if test_mode else "train"))
    
    def reset(self):
        for cb in self.callbacks:
            cb.reset()
