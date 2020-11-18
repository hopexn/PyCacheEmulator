import multiprocessing as mp
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 设置numpy输出格式
np.set_printoptions(precision=3, suppress=True)

# 啰嗦模式
verbose = False

# 记录日志
_logger: SummaryWriter = None
_mutex = mp.Lock()


# 设置logger
def init(**kwargs):
    global _logger
    global verbose
    
    verbose = kwargs.get("verbose", False)
    log_dir = os.path.join(os.path.expanduser('~/default_log'), kwargs.get("log_id", "0000"))
    os.system("mkdir -p {}".format(log_dir))
    _logger = SummaryWriter(log_dir=log_dir, flush_secs=10)


# 关闭logger
def close():
    global _logger
    with _mutex:
        _logger.flush()
        _logger.close()


# 将标量数据写到日志
def write_scalar(tag, scalar_value, global_step):
    assert _logger is not None
    with _mutex:
        _logger.add_scalar(tag, scalar_value, global_step=global_step)
        if verbose:
            print("{},{}: {}".format(int(global_step), tag, scalar_value))


# 将多个标量数据写到日志
def write_scalars(main_tag, tag_scalar_dict, global_step):
    assert _logger is not None
    with _mutex:
        _logger.add_scalars(main_tag, tag_scalar_dict, global_step=global_step)
        if verbose:
            print("{},{}: {}".format(int(global_step), main_tag, tag_scalar_dict))


def console(*msg):
    with _mutex:
        print(*msg)
