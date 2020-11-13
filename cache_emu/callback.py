import os
import threading

from torch.utils.tensorboard import SummaryWriter


class Callback:
    def __init__(self, n_step_per_episode, **kwargs):
        self.n_step_per_episode = n_step_per_episode
        
        self.i_step = 0
        self.i_episode = 0
        self.fields = kwargs
    
    def reset(self):
        self.i_step = 0
        self.i_episode = 0
    
    def on_step_end(self, **kwargs):
        self.i_step += 1
        if self.i_step % self.n_step_per_episode == 0:
            self.i_episode += 1
            self.on_episode_end(**kwargs)
    
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
        
        self.register_callback(LogCallback(**log_config, **kwargs))
    
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
    _mutex = threading.Lock()
    _writter: SummaryWriter = None
    
    def __init__(self, n_step_per_episode=1000, log_dir=None, log_id=0, main_tag="", sub_tag="",
                 verbose=False, **kwargs):
        super(LogCallback, self).__init__(n_step_per_episode, **kwargs)
        self.main_tag = main_tag
        self.sub_tag = sub_tag
        self.verbose = verbose
        
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
        self.total_hit_cnt = 0
        self.total_req_cnt = 0
        
        with LogCallback._mutex:
            if LogCallback._writter is None:
                if log_dir is None or not os.path.exists(log_dir):
                    log_dir = os.path.join(os.path.expanduser('~'), 'default_log')
                    print("Warnning: use default log_dir: {}.".format(log_dir))
                os.system("mkdir -p {}".format(log_dir))
                LogCallback._writter = SummaryWriter(
                    log_dir=os.path.join(log_dir, str(log_id)),
                    flush_secs=10)
    
    def on_step_end(self, slice_hit_cnt=0, slice_req_cnt=0, **kwargs):
        self.episode_hit_cnt += slice_hit_cnt
        self.episode_req_cnt += slice_req_cnt
        self.total_hit_cnt += slice_hit_cnt
        self.total_req_cnt += slice_req_cnt
        
        super().on_step_end(**kwargs)
    
    def on_episode_end(self, **kwargs):
        episode_hit_rate = self.episode_hit_cnt / (self.episode_req_cnt + 1e-6)
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        self._write_scalars("MHR", {self.sub_tag: mean_hit_rate})
        self._write_scalars("EHR", {self.sub_tag: episode_hit_rate})
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
    
    def _write_scalars(self, tag_prefix, scalar_dict):
        with LogCallback._mutex:
            LogCallback._writter.add_scalars(
                main_tag="{}_{}".format(tag_prefix, self.main_tag),
                tag_scalar_dict=scalar_dict,
                global_step=self.i_episode,
                walltime=self.i_episode
            )
        if self.verbose:
            with LogCallback._mutex:
                print("{},{}_{}: {}".format(self.i_episode, tag_prefix, self.main_tag, scalar_dict))
    
    def on_game_end(self, **kwargs):
        with LogCallback._mutex:
            if LogCallback._writter is not None:
                LogCallback._writter.close()
                LogCallback._writter = None
        
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        return {
            "mean_hit_rate": "{:.1f}%".format(mean_hit_rate * 100),
            "total_hit_cnt": str(self.total_hit_cnt),
            "total_req_cnt": str(self.total_req_cnt)
        }
