import os

import torch
import torch.nn.functional as F

from py_cache_emu import Callback
from py_cache_emu import torch_utils as ptu
from py_cache_emu.utils import log_utils
from py_cache_emu.utils import mp_utils as mpu
from .ewdrl import RLModel
from .kd_model import KDWeights


class HardKDCallback(Callback):
    def __init__(self, model, memory,
                 batch_size=128, interval=10, lr=0.001, alpha=0.01, weights_path="~/default_weights/",
                 k=0, mode=0, **kwargs):
        super().__init__(interval=interval)
        
        self.model: RLModel = model
        self.memory = memory
        
        self.batch_size = batch_size
        self.weights_path = os.path.expanduser(weights_path) + str(kwargs.get("rank", 0))
        os.system("mkdir -p {}".format(self.weights_path))
        self.ws: KDWeights = KDWeights(mpu.comm_size, lr, alpha, **kwargs)
        
        self.loss_fn = F.mse_loss
        self.k = k
        
        self.main_tag = kwargs.get("main_tag", "")
        self.sub_tag = kwargs.get("sub_tag", "")
        self.verbose = kwargs.get("verbose", False)
        
        self.mode = mode
    
    def on_game_begin(self, **kwargs):
        if self.weights_path is not None:
            res = self.ws.load_weights(self.weights_path, suffix="_{}".format(mpu.comm_size))
            return {"load_kd_weights": res}
    
    def on_game_end(self, **kwargs):
        if self.weights_path is not None:
            self.ws.save_weights(self.weights_path, suffix="_{}".format(mpu.comm_size))
    
    def on_episode_end(self, **kwargs):
        if self.mode == 0:
            return self.kd_mode0(**kwargs)
        elif self.mode == 1:
            return self.kd_mode1(**kwargs)
        else:
            return self.kd_mode2(**kwargs)
    
    def kd_mode0(self, **kwargs):
        batch_inputs = self.memory.sample_kd_transition(self.batch_size)
        if batch_inputs is None or len(batch_inputs) == 0:
            obs_ipts, reward_ipts, next_obs_ipts = ptu.tensor([]), ptu.tensor([]), ptu.tensor([])
            obs_opts, next_obs_opts = ptu.tensor([]), ptu.tensor([])
        else:
            obs_ipts, reward_ipts, next_obs_ipts = batch_inputs
            obs_opts = self.model.forward_distilling(obs_ipts)
        
        obs_ipts_list = mpu.all_to_all(ptu.get_numpy(obs_ipts))
        obs_opts_list = mpu.all_to_all(ptu.get_numpy(obs_opts))
        
        losses = []
        for i, (batch_inputs, batch_outputs) in enumerate(zip(obs_ipts_list, obs_opts_list)):
            loss = 0
            if len(batch_inputs) > 0:
                q_values = self.model.forward_distilling(ptu.from_numpy(batch_inputs))
                loss = self.loss_fn(q_values, ptu.from_numpy(batch_outputs))
            losses.append(loss)
        
        # 更新权重, 更新模型
        if self.k == 0:
            loss = self.ws.forward(losses)
        elif self.k > 0:
            loss = self.ws.forward_random_k(losses, self.k)
        else:
            loss = self.ws.forward_random_k_with_ref(losses, -self.k)
        
        self.ws.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.model.step()
        self.ws.step()
        self.write_ws()
        
        return {"kd_loss": loss.cpu().item()}
    
    def kd_mode1(self, **kwargs):
        batch_inputs = self.memory.sample_kd_transition(self.batch_size)
        if batch_inputs is None or len(batch_inputs) == 0:
            obs_ipts, adv_values = ptu.tensor([]), ptu.tensor([])
        else:
            obs_ipts, reward_ipts, next_obs_ipts = batch_inputs
            obs_opts = self.model.forward_distilling(obs_ipts)
            next_obs_opts = self.model.forward_distilling(next_obs_ipts)
            adv_values = reward_ipts + 0.99 * next_obs_opts - obs_opts
        
        obs_ipts_list = mpu.all_to_all(ptu.get_numpy(obs_ipts))
        adv_values_list = mpu.all_to_all(ptu.get_numpy(adv_values))
        
        losses = []
        for i, (obs_ipts, adv_values) in enumerate(zip(obs_ipts_list, adv_values_list)):
            loss = 0
            if len(obs_ipts) > 0:
                state_values = self.model.forward_distilling(ptu.from_numpy(obs_ipts))
                loss = -(ptu.from_numpy(adv_values) * state_values).mean()
            losses.append(loss)
        
        # 更新权重, 更新模型
        if self.k == 0:
            loss = self.ws.forward(losses)
        elif self.k > 0:
            loss = self.ws.forward_random_k(losses, self.k)
        else:
            loss = self.ws.forward_random_k_with_ref(losses, -self.k)
        
        self.ws.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.model.step()
        self.ws.step()
        self.write_ws()
        
        return {"kd_loss": loss.cpu().item()}
    
    def kd_mode2(self, **kwargs):
        batch_inputs = self.memory.sample_kd_transition(self.batch_size)
        if batch_inputs is None or len(batch_inputs) == 0:
            obs_ipts, reward_ipts, next_obs_ipts = ptu.tensor([]), ptu.tensor([]), ptu.tensor([])
        else:
            obs_ipts, reward_ipts, next_obs_ipts = batch_inputs
        
        obs_ipts_list = mpu.all_to_all(ptu.get_numpy(obs_ipts))
        reward_ipts_list = mpu.all_to_all(ptu.get_numpy(reward_ipts))
        next_obs_ipts_list = mpu.all_to_all(ptu.get_numpy(next_obs_ipts))
        
        losses = []
        for i, (obs_ipts, reward_ipts, next_obs_ipts) in enumerate(
                zip(obs_ipts_list, reward_ipts_list, next_obs_ipts_list)):
            
            loss = 0
            if len(obs_ipts) > 0:
                state_values = self.model.forward_distilling(ptu.from_numpy(obs_ipts))
                next_state_values = self.model.forward_distilling(ptu.from_numpy(next_obs_ipts))
                target = ptu.from_numpy(reward_ipts) + 0.99 * next_state_values
                loss = F.mse_loss(state_values, target.detach())
            
            losses.append(loss)
        
        # 更新权重, 更新模型
        if self.k == 0:
            loss = self.ws.forward(losses)
        elif self.k > 0:
            loss = self.ws.forward_random_k(losses, self.k)
        else:
            loss = self.ws.forward_random_k_with_ref(losses, -self.k)
        
        self.ws.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.model.step()
        self.ws.step()
        self.write_ws()
        
        return {"kd_loss": loss.cpu().item()}
    
    def write_ws(self, **kwargs):
        if self.i_episode % 100 == 0:
            global_step = self.i_episode / 100
            log_utils.write_scalars("{}/{}".format(self.main_tag, self.sub_tag),
                                    self.ws.get_dict(), global_step, self.verbose)
    
    def get_models(self, **kwargs):
        return [self.ws]


class SoftKDCallback(HardKDCallback):
    def __init__(self, model, memory,
                 batch_size=128, intervals=1, lr=0.01, tau=1.0,
                 weights_path=None, use_kl_div_loss=True, **kwargs):
        super().__init__(model, memory, batch_size, intervals, lr, weights_path, **kwargs)
        self.loss_fn = self.my_loss_func
        self.use_kl_div_loss = use_kl_div_loss
        self.tau = tau
    
    def my_loss_func(self, probs, log_target_probs):
        if self.use_kl_div_loss:
            # 根据kl散度定义实现
            log_probs = torch.log(probs + 1e-6)
            loss = (probs * (log_probs - log_target_probs)).mean()
        else:
            loss = - (probs * log_target_probs).mean()
        
        return loss
    
    def on_episode_end(self, **kwargs):
        batch_inputs = self.memory.sample_observations(self.batch_size)
        probs = self.model.forward_distilling(batch_inputs)
        log_probs = torch.log(probs + 1e-6)
        log_probs_softened = log_probs / self.tau
        batch_outputs = log_probs_softened
        
        batch_inputs_list = mpu.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpu.all_to_all(ptu.get_numpy(batch_outputs))
        
        losses = []
        for i, (batch_inputs, batch_outputs) in enumerate(zip(batch_inputs_list, batch_outputs_list)):
            probs = self.model(ptu.from_numpy(batch_inputs))
            loss = self.loss_fn(probs, ptu.from_numpy(batch_outputs))
            losses.append(loss)
        loss = self.ws.forward(losses)
        
        # 更新权重
        self.model.zero_grad()
        self.ws.zero_grad()
        loss.backward()
        self.model.step()
        self.ws.step()
        
        self.write_ws()
        
        return {"kd_loss": loss.cpu().item()}


def eval_callback_class(callback_class_name):
    return eval(callback_class_name)
