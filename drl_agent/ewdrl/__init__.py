from .agent.ewdnn import EWDNN
from .agent.ewdqn import EWDQN
from .agent.ewsac import EWSAC
from .agent.ewsql import EWSQL
from .core import Agent
from .model import RLModel


def eval_agent_class(agent_name):
    return eval(agent_name)
