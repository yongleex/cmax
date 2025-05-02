import numpy as np
from .slicer import 	Slicer
from .estimator import estimator


class EBIV():
    def __init__(self, cfg):
        self._c = cfg
        self.slicer = Slicer(cfg)
        self.estimator = estimator(cfg)
        
    def compute(self, events, warmstart=True):
        subsets, pos = self.slicer.split(events)
        results = np.zeros((len(subsets),2))
        for k, evts in enumerate(subsets):
            # v0 = np.array([[-0.0,0.0]]) if k==0 else results[k-1:k,:] # 热启动，用上个值作为下一个计算初值
            # v0 = v0 if warmstart else None
            v1 = self.estimator.compute(evts, v0=None)                    
            # v1 = self.estimator.compute(evts, v0=v0, debug=False)                    # 更新计算
            results[k,:] = v1[:]

        # pos = np.array(self.slicer.positions())
        vecs = np.concatenate([pos, results], axis=-1)
        return vecs