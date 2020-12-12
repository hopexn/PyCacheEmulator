import os

import torch
import torch.nn.functional as F

from py_cache_emu import Callback
from py_cache_emu import torch_utils as ptu
from py_cache_emu.utils import log_utils
from .ewdrl import RLModel
from .kd import eval_kd_mode


class HardKDCallback(Callback):
    def __init__(self, model, memory, comm,
                 batch_size=128, interval=10, sigma=1e-4, **kwargs):
        super().__init__(interval=interval, **kwargs)
        
        self.model: RLModel = model
        self.memory = memory
        self.comm = comm
        
        self.sigma = sigma
        
        self.kd_mode = int(kwargs.get("kd_mode", 0))
        self.rank = int(kwargs.get("rank", 0))
        self.data_rank = int(kwargs.get("data_rank", 0))
        self.main_tag = kwargs.get("main_tag", "")
        self.sub_tag = kwargs.get("sub_tag", "")
        self.verbose = kwargs.get("verbose", False)
        
        self.batch_size = batch_size
        self.weights_path = kwargs.get("weights_path", "~/default_weights/")
        self.weights_path += kwargs.get("weights_id", "0000")
        if self.weights_path is not None:
            self.weights_path = os.path.join(
                os.path.expanduser(self.weights_path),
                "{}".format(self.data_rank)
            )
            os.system("mkdir -p {}".format(self.weights_path))
        
        kd_class = eval_kd_mode(self.kd_mode)
        self.ws = kd_class(self.comm.comm_size, **kwargs)
        self.loss_fn = F.mse_loss
        
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
    
    def write_ws(self, **kwargs):
        if self.i_episode % 100 == 0:
            global_step = self.i_episode / 100
            log_utils.write_scalars("{}/{}".format(self.main_tag, self.sub_tag),
                                    self.ws.get_dict(), global_step, self.verbose)
    
    def write_episode_reward(self, eps_reward):
        if self.i_episode % 100 == 0:
            global_step = self.i_episode / 100
            log_utils.write_scalar("{}/{}/REWARD".format(self.main_tag, self.sub_tag), eps_reward, global_step,
                                   self.verbose)
    
    def get_models(self, **kwargs):
        return [self.ws]
    
    def on_game_begin(self, **kwargs):
        if self.weights_path is not None and self.kwargs.get("load_weights", True):
            res = self.ws.load_weights(self.weights_path)
            return {"load_kd_weights": res}
    
    def on_game_end(self, **kwargs):
        if self.weights_path is not None:
            self.ws.save_weights(self.weights_path)
    
    def on_step_end(self, slice_hit_cnt=0, slice_req_cnt=0, **kwargs):
        self.episode_hit_cnt += slice_hit_cnt
        self.episode_req_cnt += slice_req_cnt
        
        super().on_step_end(**kwargs)
    
    def on_episode_end(self, **kwargs):
        reward = self.episode_hit_cnt / (self.episode_req_cnt + 1e-6)
        reward = ptu.float_tensor(reward).unsqueeze(0)
        
        # self.write_episode_reward(reward)
        
        self.episode_hit_cnt = 0
        self.episode_req_cnt = 0
        
        sample_ipts, sample_rewards = self.memory.sample_kd_samples(self.batch_size)
        if sample_ipts is None or len(sample_ipts) == 0:
            sample_ipts, sample_opts = ptu.tensor([]), ptu.tensor([])
            sample_rewards = ptu.tensor(0).unsqueeze(0)
        else:
            with torch.no_grad():
                sample_ipts += self.sigma * torch.randn_like(sample_ipts)
                sample_opts = self.model.forward_distilling(sample_ipts)
                sample_rewards = sample_rewards.mean().unsqueeze(0)
        
        sample_ipts_list = self.comm.all_to_all(ptu.get_numpy(sample_ipts), self.rank)
        sample_rewards_list = self.comm.all_to_all(ptu.get_numpy(sample_rewards), self.rank)
        rewards_list = self.comm.all_to_all(ptu.get_numpy(reward), self.rank)
        sample_opts_list = self.comm.all_to_all(ptu.get_numpy(sample_opts), self.rank)
        losses = []
        for i, (sample_ipts, sample_opts) in enumerate(zip(sample_ipts_list, sample_opts_list)):
            loss = ptu.float_tensor(0, requires_grad=True)
            if len(sample_ipts) > 0:
                preds = self.model.forward_distilling(ptu.from_numpy(sample_ipts))
                loss = self.loss_fn(preds, ptu.from_numpy(sample_opts))
            losses.append(loss.unsqueeze(0))
        
        sample_rewards_list = ptu.tensor(sample_rewards_list).squeeze(-1)
        rewards_list = ptu.tensor(rewards_list).squeeze(-1)
        # 更新权重, 更新模型
        loss = self.ws.forward(torch.cat(losses), reward=reward,
                               rewards=rewards_list, sample_rewards=sample_rewards_list)
        
        self.ws.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.model.step()
        self.ws.step()
        
        self.write_ws()
        
        if (self.i_episode + 1) % 500 == 0 and self.kwargs.get("tmp_save", True):
            self.ws.save_weights(self.weights_path)
        
        return {"kd_loss": loss.cpu().item()}


class SoftKDCallback(HardKDCallback):
    def __init__(self, model, memory, batch_size=128, interval=10, lr=0.001, sigma=1e-2, kd_tau=1.0,
                 use_kl_div_loss=True, **kwargs):
        super().__init__(model, memory, batch_size, interval, lr, sigma, **kwargs)
        
        self.loss_fn = self.my_loss_func
        self.use_kl_div_loss = use_kl_div_loss
        self.kd_tau = kd_tau
    
    def my_loss_func(self, logits, target_logits):
        probs = (logits / self.kd_tau).softmax(dim=-1)
        with torch.no_grad():
            log_target_probs = (target_logits / self.kd_tau).softmax(dim=-1).log()
        
        if self.use_kl_div_loss:
            # 根据kl散度定义实现
            log_probs = torch.log(probs)
            loss = (probs * (log_probs - log_target_probs)).mean()
        else:
            loss = - (probs * log_target_probs).mean()
        
        return loss


def eval_callback_class(callback_class_name):
    return eval(callback_class_name)
