class Agent:
    def __init__(self, content_dim, feature_dim, **kwargs):
        self.content_dim = content_dim
        self.feature_dim = feature_dim
    
    def forward(self, observation):
        raise NotImplementedError()
    
    def backward(self, observation, action, reward, next_observation):
        raise NotImplementedError()
    
    def save_weights(self, path, prefix="", suffix=""):
        raise NotImplementedError()
    
    def load_weights(self, path, prefix="", suffix=""):
        raise NotImplementedError()
    
    def get_models(self):
        raise NotImplementedError()
    
    def get_distilling_model(self):
        raise NotImplementedError()
    
    def get_distilling_memory(self):
        raise NotImplementedError()


class QPolicy:
    def select_action(self, q_values):
        raise NotImplementedError()
    
    def backward(self, **kwargs):
        pass


class CallBack:
    def __init__(self, **kwargs):
        pass
    
    def on_step_end(self, **kwargs):
        pass
    
    def on_episode_end(self, **kwargs):
        pass
    
    def on_game_begin(self, **kwargs):
        pass
    
    def on_game_end(self, **kwargs):
        pass
