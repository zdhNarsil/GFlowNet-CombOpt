# GFlowNet-CombOpt
Pytorch implementation for our paper 

[Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets](https://arxiv.org/abs/2305.17010).

[Dinghuai Zhang](https://zdhnarsil.github.io/), [Hanjun Dai](https://hanjun-dai.github.io/), Nikolay Malkin, Aaron Courville, [Yoshua Bengio](https://yoshuabengio.org/), [Ling Pan](https://ling-pan.github.io/).

<!-- <p align="center"> -->
<img src="https://s1.ax1x.com/2023/05/30/p9jE7P1.png" border="0" width=60% class="center" />
<!-- </p> -->

We formulate a set of graph combinatorial optimization problems as sequential decision-making sampling problems,
and design efficient GFlowNet algorithms to tackle them.
 

## Dependency

```bash
pip install hydra-core==1.1.0 omegaconf submitit hydra-submitit-launcher
pip install dgl==0.6.1
```

## Data generation

```bash
cd data/
python rbgraph_generator.py --num_graph 4000 --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --save_dir rb200-300/test  
```

## Training

```bash
cd gflownet/
python main.py input=rb200-300 alg=fl bsit=8
```