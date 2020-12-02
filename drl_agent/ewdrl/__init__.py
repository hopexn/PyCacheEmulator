from .agent.ewdnn import EWDNN
from .agent.ewdqn import EWDQN
from .core import Agent
from .model import RLModel


def eval_agent_class(agent_name):
    return eval(agent_name)
