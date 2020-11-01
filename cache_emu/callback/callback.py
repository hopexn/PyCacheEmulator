class Callback:
    def __init__(self):
        self.test_mode = False
    
    def reset(self):
        pass
    
    def on_step_end(self):
        pass
    
    def on_episode_end(self, **kwargs):
        pass
    
    def on_game_begin(self, **kwargs):
        pass
    
    def on_game_end(self, **kwargs):
        pass
    
    def switch_mode(self, test_mode=False):
        self.test_mode = test_mode
