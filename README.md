# GFlowNet-CombOpt
Pytorch implementation for our paper "Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets".

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

## Dependency

```bash
pip install hydra-core==1.1.0 omegaconf submitit hydra-submitit-launcher
pip install dgl==0.6.1
```