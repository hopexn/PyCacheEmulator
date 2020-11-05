from tqdm import trange

from .callback import Callback


class TqdmCallback(Callback):
    def __init__(self, n_episodes):
        super().__init__()
        self.n_episodes = n_episodes
        self.trange = trange(self.n_episodes)
    
    def reset(self):
        self.trange.clear()
        self.trange.reset()
    
    def on_episode_end(self, description={}, postfix={}, **kwargs):
        self.trange.set_description(description)
        self.trange.set_postfix(postfix)
        self.trange.update()
    
    def on_game_end(self, **kwargs):
        self.trange.close()
        self.trange.clear()
