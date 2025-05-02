# CMax
Our implementation of contrast maximization for EBV

## Installation and Usage
- Install
```
git clone git@github.com:yongleex/cmax.git
cd cmax
# conda activate your_python_env
pip install .  
```

- Usage
```
from cmax.config import config
from cmax.ebiv import EBIV

cfg = config()        # Default config
cfg.grid_sz=[256,256] # or, you can specify it 

ebiv = EBIV(cfg)

events = ...   # N*4
vecs = ebiv.compute(events) # M*4 for x,y,vx,vy
```

Example
- [Test.ipynb](https://github.com/yongleex/cmax/blob/main/tests/Test.ipynb)


