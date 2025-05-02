from abc import ABC, abstractmethod # Abstract Class for uniform interface

import numpy as np
import torch
import torch.optim as optim

class EstimatorABC(ABC): # 事件测量
    def __init__(self, cfg):
        self._c = cfg

    @abstractmethod
    def compute(self, evts, v0=None):
        """Return a np.array(1*2) velocity vector
        """
        pass


class EstimatorCMax(EstimatorABC):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute(self, evts, v0=None):
        evts = torch.tensor(evts).to(self.device)
        v0 = torch.tensor([[0.0, 0.0]],device=self.device, requires_grad=True) 
        
        # optimizer = optim.SGD([v0], lr=self._c.lr)  
        # optimizer = optim.Adamax([v0], lr=self._c.lr)  
        optimizer = optim.Adam([v0], lr=self._c.lr)  
        # optimizer = optim.AdamW([v0], lr=self._c.lr)  
        # optimizer = optim.Adagrad([v0], lr=self._c.lr)  
        # optimizer = optim.RMSprop([v0], lr=self._c.lr)  
        
       
        # 优化过程
        for epoch in range(self._c.MaxIter):
            optimizer.zero_grad()  # 清除之前的梯度
            loss = self.iwe_var(evts, v0)    # 计算目标函数值
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新参数
            
            # 打印进度
            if epoch % 8 == 0 and self._c.debug:
                print(f'Epoch {epoch}: v0 = {v0.cpu().detach().numpy()}, f(x) = {loss.item()}')
        return v0.cpu().detach().numpy()

    def iwe_var(self, evts, v0):
        """
        evts: N*4
        v0: M*2
        """
        x, y, t, p = torch.split(evts, 1, dim=1)
        x, y, t, p = x.reshape(1,-1), y.reshape(1,-1), t.reshape(1,-1), p.reshape(1, -1)
        vx, vy = torch.split(v0, 1, dim=1)
        x_prime, y_prime = x-t*vx, y-t*vy
        
        x_min, y_min = torch.min(x_prime,dim=-1,keepdim=True)[0], torch.min(y_prime,dim=-1,keepdim=True)[0]
        x_p, y_p = x_prime-x_min, y_prime-y_min
        x_p, y_p = x_p.unsqueeze(dim=1), y_p.unsqueeze(dim=1)
        # print(x_prime.shape, x_p.shape)
    
        # 生成坐标序列（确保整数类型）
        px = torch.arange(0, self._c.IWE_sz[0], dtype=torch.long, device=self.device)
        py = torch.arange(0, self._c.IWE_sz[1], dtype=torch.long, device=self.device)
        x_grid, y_grid = torch.meshgrid(px, py, indexing='xy')
        x_grid, y_grid = x_grid.flatten().reshape([1,-1,1]),y_grid.flatten().reshape([1,-1,1])
    
        # 计算距离
        dx, dy = x_p-x_grid, y_p-y_grid
        weight = torch.exp(-0.5*dx*dx/self._c.sigma2)*torch.exp(-0.5*dy*dy/self._c.sigma2)
        # print("weight", weight.shape)
        iwe = torch.sum(weight, dim=-1, keepdim=True)
        iwe_mean = torch.mean(iwe,dim=-2,keepdim=True)
        # print(weight.shape, iwe.shape, iwe_mean.shape)
        # vars = -torch.log(torch.mean(torch.square(iwe-iwe_mean),dim=-2,keepdim=True))
        vars = -torch.mean(torch.square(iwe-iwe_mean),dim=-2,keepdim=True)
        return vars


class EstimatorPCM(EstimatorABC):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute(self, evts, v0=None):
        evts = torch.tensor(evts).to(self.device)
        v0 = torch.tensor([[0.0, 0.0]],device=self.device, requires_grad=True) 
        
        # optimizer = optim.SGD([v0], lr=0.5)  # 随机梯度下降，学习率0.1
        # optimizer = optim.Adamax([v0], lr=0.2)  # 随机梯度下降，学习率0.1
        optimizer = optim.Adam([v0], lr=self._c.lr)  # 随机梯度下降，学习率0.1
        # optimizer = optim.AdamW([v0], lr=0.1)  # 随机梯度下降，学习率0.1
        # optimizer = optim.Adagrad([v0], lr=0.2)  # 随机梯度下降，学习率0.1
        # optimizer = optim.RMSprop([v0], lr=0.1)  
        
        # 
        # 优化过程
        for epoch in range(self._c.MaxIter):
            optimizer.zero_grad()  # 清除之前的梯度
            loss = self.pcm_loss(evts, v0)    # 计算目标函数值
            loss.backward()        # 反向传播计算梯度
            optimizer.step()       # 更新参数
            
            # 打印进度
            if epoch % 1000 == 0 and self._c.debug:
                print(f'Epoch {epoch}: v0 = {v0.cpu().detach().numpy()}, f(x) = {loss.item()}')
        return v0.cpu().detach().numpy()

    def pcm_loss(self, evts, v0):
        """
        evts: N*4
        v0: M*2
        """
        x, y, t, p = torch.split(evts, 1, dim=1)
        xij = x.reshape(1,-1) - x.reshape(-1,1) 
        yij = y.reshape(1,-1) - y.reshape(-1,1) 
        tij = t.reshape(1,-1) - t.reshape(-1,1) 
        # x, y, t, p = x.reshape(1,-1), y.reshape(1,-1), t.reshape(1,-1), p.reshape(1, -1)
        vx, vy = torch.split(v0, 1, dim=1)

        ex, ey = vx*tij-xij, vy*tij-yij
        loss = -torch.mean(torch.exp(-0.5*ex*ex/self._c.sigma2-0.5*ey*ey/self._c.sigma2))
        return loss


def estimator(cfg):
    return eval(cfg.estimator)(cfg)