# PDE Learning
## 1D Burgers' Equation
$$
\partial_{t}u(t,x)+\partial_{x}(u^{2}(t,x)/2)=\nu/\pi\partial_{xx}u(t,x),\quad x\in(0,1),t\in(0,2]\\
u(0,x)=u_{0}(x),\quad x\in(0,1)
$$

Three models:
- FNO 2D
- PINO
- U-Net 1D
- U-Net 2D

```
python main.py --config_path ./config/burgers/pino.yaml
```

## Navier-Stokes Equation
$$
\partial_{t}w(x,t)+u(x,t)\cdot\nabla w(x,t)=\nu\Delta w(x,t)+f(x),\quad x\in(0,1)^{2},t\in(0,T]\\
\nabla\cdot u(x,t)=0,\quad x\in(0,1)^{2},t\in(0,T]\\
w(x,0)=w_{0}(x),\quad x\in(0,1)^{2},t\in(0,T]
$$

Two models:
- FNO 3D
- PINO
```
python main.py --config_path ./config/km_flow/pino.yaml
```