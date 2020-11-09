import os
import threading

import enlighten
from torch.utils.tensorboard import SummaryWriter


class Callback:
    def __init__(self, **kwargs):
        self.i_step = 0
        self.i_episode = 0
        
        self.test_mode = False
    
    def reset(self):
        self.i_step = 0
        self.i_episode = 0
    
    def on_step_end(self, **kwargs):
        self.i_step += 1
    
    def on_episode_end(self, **kwargs):
        self.i_episode += 1
    
    def on_game_begin(self, **kwargs):
        return {}
    
    def on_game_end(self, **kwargs):
        return {}
    
    def switch_mode(self, test_mode=False):
        self.test_mode = test_mode


class CallbackManager:
    def __init__(self, total_steps, n_episode_steps: int = 1000, callbacks=[],
                 log_config={}, **kwargs):
        self.callbacks = []
        self.total_steps = total_steps
        self.n_episode_steps = n_episode_steps
        self.n_episodes = int(self.total_steps / self.n_episode_steps)
        self.step_count = 0
        
        for cb in callbacks:
            assert isinstance(cb, Callback)
            self.callbacks.append(cb)
        
        self.register_callback(LogCallback(n_episodes=self.n_episodes, **log_config, **kwargs))
    
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def reset(self):
        for cb in self.callbacks:
            cb.reset()
    
    def on_step_end(self, **kwargs):
        self.step_count += 1
        
        info = {}
        
        for cb in self.callbacks:
            cb.on_step_end(**kwargs)
        
        if self.step_count % self.n_episode_steps == 0:
            res = self.on_episode_end(**kwargs)
            if res is not None:
                info.update(res)
        
        return info
    
    def on_episode_end(self, **kwargs):
        info = {}
        
        for cb in self.callbacks:
            res = cb.on_episode_end(**kwargs)
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
    
    def switch_mode(self, test_mode=False):
        print("Mode: {}".format("test" if test_mode else "train"))
        for cb in self.callbacks:
            cb.switch_mode(test_mode)


class LogCallback(Callback):
    _manager = enlighten.get_manager(
        bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]  {env_info} "
    )
    _mutex = threading.Lock()
    
    @staticmethod
    def dict2str(d: dict):
        str_list = []
        for k, v in d.items():
            tpl = "{}:{:.3f}" if isinstance(v, float) else "{}:{}"
            str_list.append(tpl.format(k, v))
        return ", ".join(str_list)
    
    def __init__(self, n_episodes, tag_dict={}, log_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.n_episodes = n_episodes
        self.tag_dict = tag_dict
        self.log_dir = log_dir
        
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
        self.total_hit_cnt = 0
        self.total_req_cnt = 0
        
        if self.log_dir is None:
            self.log_dir = os.path.join(os.path.expanduser('~'), 'default_log')
        
        if not os.path.exists(self.log_dir):
            os.system("mkdir -p {}".format(self.log_dir))
        
        self.summary_writter = None
        if os.path.exists(self.log_dir):
            self.summary_writter = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        self.algo_name = kwargs.get("algo_name", "null")
        self.main_tag = kwargs.get("tag_name", "null")
        with LogCallback._mutex:
            self.bar = LogCallback._manager.counter(
                total=self.n_episodes, desc="{}_{}".format(self.algo_name, self.main_tag), env_info=""
            )
    
    def reset(self):
        with LogCallback._mutex:
            self.bar.clear()
    
    def on_step_end(self, step_hit_cnt=0, step_req_cnt=0, **kwargs):
        super().on_step_end(**kwargs)
        self.episode_hit_cnt += step_hit_cnt
        self.episode_req_cnt += step_req_cnt
        self.total_hit_cnt += step_hit_cnt
        self.total_req_cnt += step_req_cnt
    
    def on_episode_end(self, env_info={}, **kwargs):
        super().on_episode_end(**kwargs)
        env_info.update(self._get_hit_rate_info())
        
        if self.summary_writter is not None:
            for k, v in env_info.items():
                self.summary_writter.add_scalars(
                    main_tag="{}_{}".format(k, self.main_tag),
                    tag_scalar_dict={self.algo_name: env_info[k]},
                    global_step=self.i_episode)
        
        env_info_str = self.dict2str(env_info)
        with LogCallback._mutex:
            self.bar.update(env_info=env_info_str, **kwargs)
    
    def on_game_end(self, **kwargs):
        if self.summary_writter is not None:
            self.summary_writter.close()
        
        with LogCallback._mutex:
            self.bar.close()
        
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        return {
            "mean_hit_rate": "{:.1f}%".format(mean_hit_rate * 100),
            "total_hit_cnt": str(self.total_hit_cnt),
            "total_req_cnt": str(self.total_req_cnt)
        }
    
    def _get_hit_rate_info(self):
        mean_hit_rate = self.total_hit_cnt / (self.total_req_cnt + 1e-6)
        episode_hit_rate = self.episode_hit_cnt / (self.episode_req_cnt + 1e-6)
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
        info = {"MHR": mean_hit_rate, "EHR": episode_hit_rate}
        return info
