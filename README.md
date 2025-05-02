# CMax
Our implementation of contrast maximization for EBV

## Install and usage
Installation
```
git clone 
cd cmax
pip install .
```

Usage
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
[Test.ipynb]()


