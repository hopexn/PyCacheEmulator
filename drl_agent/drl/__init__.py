from .agent.ewdnn import EWDNN
from .agent.ewdqn import EWDQN
from .agent.ewsac import EWSAC
from .agent.ewsql import EWSQL
from .agent.ewsql2 import EWSQL2
from .core import Agent
from .model import RLModel
from .utils import numpy_utils as npu
from .utils import torch_utils as ptu


def eval_agent_class(agent_name):
    return eval(agent_name)
