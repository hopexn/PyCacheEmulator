from .agent import Agent
from .runner import CacheRunner
from .runner import LfuCacheRunner, LruCacheRunner, RandomCacheRunner
from .runner import OgdLfuCacheRunner, OgdLruCacheRunner, OgdOptCacheRunner


def eval_runner(runner_name):
    return eval(runner_name)
