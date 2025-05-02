class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    cfg = AttrDict()
    cfg.name = "CMax config"

    # splicer
    cfg.grid_sz = [512,512]
    cfg.win_sz = [40,40]
    cfg.win_step = [16,16]
    cfg.time_interval = [None, None]

    # CMax estimator
    cfg.estimator="EstimatorCMax"
    cfg.IWE_sz = [32,32]
    cfg.sigma2 = 0.25
    cfg.MaxIter = 48
    cfg.lr = 0.1 # learning rate
    cfg.debug = False

    # # PCM estimator
    # cfg.estimator="EstimatorPCM"

    return cfg
