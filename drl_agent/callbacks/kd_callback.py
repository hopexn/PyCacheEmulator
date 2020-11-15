import gym
import numpy as np
import torch
import torch.nn.functional as F

from cache_emu import Callback
from ..drl import RLModel
from ..drl import ptu
from ..kd_model import KDModel


class HardKDCallback(Callback):
    def __init__(self, model, memory,
                 batch_size=128, interval=1, lr=0.01, tau=1.0, weights_path=None,
                 k=2, alpha=0.01, **kwargs):
        super().__init__(interval=interval)
        
        self.model: RLModel = model
        self.memory = memory
        
        self.batch_size = batch_size
        self.lr = lr
        self.weights_path = weights_path
        self.tau = tau
        self.ws = KDModel(mpi_utils.comm_size, lr, alpha=alpha)
        
        self.loss_fn = F.mse_loss
        self.k = k
    
    def on_game_begin(self):
        if self.weights_path is not None:
            res = self.ws.load_weights(self.weights_path, suffix="_{}".format(mpi_utils.comm_size))
            return {"load_kd_weights": res}
    
    def on_game_end(self):
        if self.weights_path is not None:
            self.ws.save_weights(self.weights_path, suffix="_{}".format(mpi_utils.comm_size))
    
    def on_episode_end(self):
        self.i_episode += 1
        
        batch_inputs = self.memory.sample_observations(self.batch_size)
        if len(batch_inputs) > 0:
            batch_outputs = self.model.forward_distilling(batch_inputs)
        else:
            batch_outputs = torch.tensor([])
        
        batch_inputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_outputs))
        
        losses = []
        for i, (batch_inputs, batch_outputs) in enumerate(zip(batch_inputs_list, batch_outputs_list)):
            loss = 0
            if len(batch_inputs) > 0:
                q_values = self.model.forward_distilling(ptu.from_numpy(batch_inputs))
                loss = self.loss_fn(q_values, ptu.from_numpy(batch_outputs))
            losses.append(loss)
        
        # 更新权重, 更新模型
        loss = self.ws.forward(losses)
        # self.ws.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.model.step()
        # self.ws.step()
        
        log_utils.write_ws(self.ws.get_dict(), self.i_episode)
        
        return {"kd_loss": loss.cpu().item()}
    
    def get_models(self):
        return [self.ws]


class SoftKDCallback(HardKDCallback):
    def __init__(self, model, memory,
                 batch_size=128, intervals=1, lr=0.01, tau=1.0,
                 weights_path=None, use_kl_div_loss=True, **kwargs):
        super().__init__(model, memory, batch_size, intervals, lr, tau, weights_path, **kwargs)
        self.loss_fn = self.my_loss_func
        self.use_kl_div_loss = use_kl_div_loss
    
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
    
    def on_episode_end(self):
        self.i_episode += 1
        if self.test_mode:
            return
        
        batch_inputs = self.memory.sample_observations(self.batch_size)
        probs = self.model.forward_distilling(batch_inputs)
        log_probs = torch.log(probs + 1e-6)
        log_probs_softened = log_probs / self.tau
        batch_outputs = log_probs_softened
        
        batch_inputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_outputs))
        
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
        
        log_utils.write_ws(self.ws.get_dict(), self.i_episode)
        
        return {"kd_loss": loss.cpu().item()}


class RandomKDCallback(HardKDCallback):
    def __init__(self, model, memory, batch_size=128, intervals=1, lr=0.01, tau=1.0, weights_path=None, k=2, **kwargs):
        super().__init__(model, memory, batch_size, intervals, lr, tau, weights_path, k, **kwargs)
        
        self.observation_space = gym.spaces.box.Box(
            low=np.zeros(shape=(batch_size, model.content_dim, model.feature_dim)),
            high=np.ones(shape=(batch_size, model.content_dim, model.feature_dim)),
        )
    
    def on_episode_end(self):
        self.i_episode += 1
        if self.test_mode:
            return
        
        batch_inputs = ptu.from_numpy(self.observation_space.sample())
        batch_outputs = self.model.forward_distilling(batch_inputs)
        
        batch_inputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_inputs))
        batch_outputs_list = mpi_utils.all_to_all(ptu.get_numpy(batch_outputs))
        
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
        
        log_utils.write_ws(self.ws.get_dict(), self.i_episode)
        
        return {"kd_loss": loss.cpu().item()}
