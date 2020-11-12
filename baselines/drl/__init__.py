from .agent.ewdqn import EWDQN


def eval_agent_class(agent_name):
    return eval(agent_name)
