from abc import ABC, abstractmethod # Abstract Class for uniform interface
import numpy as np


class SlicerABC(ABC): # 事件数据分块
    def __init__(self, cfg):
        self._c = cfg

    @abstractmethod
    def split(self, events):
        """ Return a list of evts, B*4 np.array
            Return the positions of all subsets
        """
        pass


class Slicer(SlicerABC):
    def split(self, events):
        # 打格子
        s0 = np.arange(0,self._c.grid_sz[0]-self._c.win_sz[0]+1,self._c.win_step[0])
        s1 = np.arange(0,self._c.grid_sz[1]-self._c.win_sz[1]+1,self._c.win_step[1])

        event_subsets = []
        poses = []
        for x_min in s0:
            for y_min in s1:
                mask = np.ones(events.shape[0], dtype=bool)
                mask &= (events[:, 0] >= x_min) & (events[:, 0] < x_min+self._c.win_sz[0]) # x component
                mask &= (events[:, 1] >= y_min) & (events[:, 1] < y_min+self._c.win_sz[1]) # y component
                if self._c.time_interval[0] is not None:                           # t component
                    mask &= (events[:, 2] >= self._c.time_interval[0]) 
                if self._c.time_interval[1] is not None:
                    mask &= (events[:, 2] < self._c.time_interval[1]) 

                # TODO polarity component
                subset = events[mask]
                event_subsets.append(subset)
                pos = [x_min+self._c.win_sz[0]/2.0, y_min+self._c.win_sz[1]/2.0]
                poses.append(pos)
        poses = np.array(poses)    
        return event_subsets, poses