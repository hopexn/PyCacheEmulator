from .drl.utils import torch_utils
from .rl_runner import RlCacheRunner
from .runners import CacheRunner, RandomCacheRunner, LfuCacheRunner, LruCacheRunner
from .runners import OgdLruCacheRunner, OgdLfuCacheRunner, OgdOptCacheRunner


def eval_runner(runner_name):
    return eval(runner_name)
