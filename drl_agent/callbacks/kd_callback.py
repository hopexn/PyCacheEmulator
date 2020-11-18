import gym
import numpy as np
import torch
import torch.nn.functional as F

from cache_emu import Callback
from cache_emu import torch_utils as ptu
from cache_emu.utils import log_utils
from cache_emu.utils import mp_utils as mpu
from ..ewdrl import RLModel
from ..kd_model import KDWeights


class HardKDCallback(Callback):
    def __init__(self, model, memory,
                 batch_size=128, interval=10, lr=0.01, weights_path=None,
                 k=0, alpha=0.01, **kwargs):
        super().__init__(interval=interval)
        
        self.model: RLModel = model
        self.memory = memory
        
        self.batch_size = batch_size
        self.lr = lr
        self.weights_path = weights_path
        self.ws = KDWeights(mpu.comm_size, lr, alpha=alpha, **kwargs)
        
        self.main_tag = kwargs.get("main_tag", "")
        self.sub_tag = kwargs.get("sub_tag", "")
        
        self.loss_fn = F.mse_loss
        self.k = k
        self.verbose = kwargs.get("verbose", False)
    
    def on_game_begin(self, **kwargs):
        if self.weights_path is not None:
            res = self.ws.load_weights(self.weights_path, suffix="_{}".format(mpu.comm_size))
            return {"load_kd_weights": res}
    
    def on_game_end(self, **kwargs):
        if self.weights_path is not None:
            self.ws.save_weights(self.weights_path, suffix="_{}".format(mpu.comm_size))
    
    def on_episode_end(self, **kwargs):
        batch_inputs = self.memory.sample_observations(self.batch_size)
        if batch_inputs is None or len(batch_inputs) == 0:
            batch_inputs = torch.tensor([])
            batch_outputs = torch.tensor([])
        else:
            batch_outputs = self.model.forward_distilling(batch_inputs)
        
        batch_inputs_list = mpu.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpu.all_to_all(ptu.get_numpy(batch_outputs))
        
        losses = []
        for i, (batch_inputs, batch_outputs) in enumerate(zip(batch_inputs_list, batch_outputs_list)):
            loss = 0
            if len(batch_inputs) > 0:
                q_values = self.model.forward_distilling(ptu.from_numpy(batch_inputs))
                loss = self.loss_fn(q_values, ptu.from_numpy(batch_outputs))
            losses.append(loss)
        
        # 更新权重, 更新模型
        if self.k <= 0:
            loss = self.ws.forward(losses)
        else:
            loss = self.ws.forward_topk(losses, self.k)
        
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
            # 使用pytorch的kl_div
            # target_probs = torch.exp(log_target_probs)
            # loss = torch.nn.functional.kl_div(probs, target_probs, reduction="batchmean")
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


class RandomKDCallback(HardKDCallback):
    def __init__(self, model, memory, batch_size=128, intervals=1, lr=0.01, tau=1.0, weights_path=None, k=2, **kwargs):
        super().__init__(model, memory, batch_size, intervals, lr, tau, weights_path, k, **kwargs)
        
        self.observation_space = gym.spaces.box.Box(
            low=np.zeros(shape=(batch_size, model.content_dim, model.feature_dim)),
            high=np.ones(shape=(batch_size, model.content_dim, model.feature_dim)),
        )
    
    def on_episode_end(self, **kwargs):
        batch_inputs = ptu.from_numpy(self.observation_space.sample())
        batch_outputs = self.model.forward_distilling(batch_inputs)
        
        batch_inputs_list = mpu.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpu.all_to_all(ptu.get_numpy(batch_outputs))
        
        losses = []
        for i, (batch_inputs, batch_outputs) in enumerate(zip(batch_inputs_list, batch_outputs_list)):
            q_values = self.model.forward_distilling(ptu.from_numpy(batch_inputs))
            loss = self.loss_fn(q_values, ptu.from_numpy(batch_outputs))
            losses.append(loss)
        
        # 更新模型
        loss2 = self.ws.forward2(losses, self.k)
        self.model.zero_grad()
        loss2.backward(retain_graph=True)
        self.model.step()
        
        # 更新权重
        loss = self.ws.forward(losses)
        self.ws.zero_grad()
        loss.backward()
        self.ws.step()
        
        self.write_ws()
        
        return {"kd_loss": loss.cpu().item()}
