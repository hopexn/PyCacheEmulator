class Callback:
    def __init__(self):
        self.test_mode = False
    
    def on_step_end(self):
        pass
    
    def on_episode_end(self):
        pass
    
    def on_game_begin(self):
        pass
    
    def on_game_end(self):
        pass
    
    def switch_mode(self, test_mode=False):
        self.test_mode = test_mode
    
    def reset(self):
        pass


class CallbackManager:
    def __init__(self, num_episode_steps: int):
        self.callbacks = []
        self.n_episode_steps = num_episode_steps
        self.step_count = 0
        
        assert (self.n_episode_steps > 0)
    
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_step_end(self):
        self.step_count += 1
        
        info = {}
        
        for cb in self.callbacks:
            cb.on_step_end()
        
        if self.step_count % self.n_episode_steps == 0:
            for cb in self.callbacks:
                res = cb.on_episode_end()
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
    
    def reset(self):
        for cb in self.callbacks:
            cb.reset()
