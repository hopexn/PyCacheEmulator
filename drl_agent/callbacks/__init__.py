from .kd_callback import HardKDCallback, SoftKDCallback, RandomKDCallback


def eval_callback_class(callback_class_name):
    return eval(callback_class_name)
