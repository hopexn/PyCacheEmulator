import os
import threading

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 设置numpy输出格式
np.set_printoptions(precision=3, suppress=True)

# 啰嗦模式
verbose = False

# 记录日志
_logger: SummaryWriter = None

# 互斥锁
_mutex = threading.Lock()


# 设置logger
def setup_logger(**kwargs):
    global _logger
    global verbose
    
    with _mutex:
        if _logger is not None:
            return
        
        log_dir = os.path.join(
            kwargs.get("log_dir", os.path.expanduser('~/default_log')),
            kwargs.get("log_id", "0")
        )
        os.system("mkdir -p {}".format(log_dir))
        print("Log dir: {}".format(log_dir))
        
        _logger = SummaryWriter(log_dir=log_dir, flush_secs=10)
        verbose = kwargs.get("verbose", False)


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
        _logger.add_scalar(tag, scalar_value, global_step=global_step, walltime=global_step)
        if verbose:
            print("{},{}: {}".format(global_step, tag, scalar_value))


# 将多个标量数据写到日志
def write_scalars(main_tag, tag_scalar_dict, global_step):
    assert _logger is not None
    with _mutex:
        _logger.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, walltime=global_step)
        if verbose:
            print("{},{}: {}".format(global_step, main_tag, tag_scalar_dict))


def console(*msg):
    with _mutex:
        print(*msg)
