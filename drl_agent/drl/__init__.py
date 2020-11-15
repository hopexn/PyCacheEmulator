from .agent.ewdnn import EWDNN
from .agent.ewdqn import EWDQN
from .agent.ewsac import EWSAC
from .agent.ewsql import EWSQL
from .agent.ewsql2 import EWSQL2


def eval_agent_class(agent_name):
    return eval(agent_name)
