from .kd_model1 import KDWeights1
from .kd_model2 import KDWeights2
from .kd_model3 import KDWeights3
from .kd_model4 import KDWeights4
from .kd_model5 import KDWeights5


def eval_kd_mode(kd_mode=0):
    return eval("KDWeights{}".format(kd_mode))
