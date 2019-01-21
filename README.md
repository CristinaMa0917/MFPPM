# MFPPM
Memory Focused Proximal Policy Method for Adaptive Biped Locomotion

- Attention mechanisem and RNN are introduced into PPO ,which makes a superior performence in control of biped robot walking. (So it's aslo called AM-RPPO)
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/%E5%9B%BE%E7%89%87%201%E7%9A%84%E5%89%AF%E6%9C%AC.png"/></div>

- Platform
  - Torch
  - OpenAI gym(four robots: BipedalWalker-V2, BipedalWalkerHardcore-v2, Humanoid-V2, Walker2d-V2)
  - DDPG (refer to Morvan's)
  - PPO (Openai Baseline)
  - RDPG (refer to Doo Re Song's)

- Performance
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/BipedalWalker-5k.png"/></div>
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/Humanoid-100k.png"/></div>
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/Walker2d-7k.png"/></div>

- Simulations
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/bw.gif"/></div>
<div align=center><img width="700" height="400" src="https://github.com/CristinaMa0917/MFPPM/blob/master/figures/bwhc.gif"/></div>
