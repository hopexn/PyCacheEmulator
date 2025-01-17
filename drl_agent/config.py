DEFAULT_DRL_FEATURE_CONFIG = {
    'swf_w_lens': [1, 6, 30, 60, 180, 360, 720, 1440, 2880, 8640, 17280, 25920, 34560, 43200, 51840, 60480]
}

DEFAULT_DRL_AGENT_CONFIG = {
    'class_name': 'EWDQN',
    'kwargs'    : {}
}

DEFAULT_DISTILLING_CONFIG = {
    'class_name': "HardKDCallback",
    'batch_size': 128,
    'tau'       : 0.2,
    'lr'        : 0.01,
    'interval'  : 10,
    'min_entropy_ratio': 0.1
}
