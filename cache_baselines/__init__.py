from .agent import Agent
from .runner import CacheRunner
from .runner import LfuCacheRunner, LruCacheRunner, RandomCacheRunner
from .runner import OgdLfuCacheRunner, OgdLruCacheRunner, OgdOptCacheRunner
from .runner import SwfCacheRunner, EwdqnCacheRunner
from .ewdrl.utils import torch_utils

def eval_runner(runner_name):
    return eval(runner_name)
