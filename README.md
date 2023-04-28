# CS598: Deep Learning for Healthcare Reproducibility Project

## Project Overview
The purpose of this project is to reproduce the research in:

Variationally regularized graph-based representation learning for electronic health records.

### General Problem
EHR data contains many medical codes and concepts
so the data is very sparse. Building graphs
from the data can be challenging and computationally
expensive. The paper aims to improve the
graph structure learning of EHR data by regularizing
node representations and adaptively learning
connections between medical codes (Zhu, W., & Razavian, N. 2021, April).

### New Approach
The new approach taken by the paper is to define
an architecture for building graphs of EHR data
that can be generalized on different datasets. The
goal is to learn the structure without explicit links
so the model is easier to apply to different datasets.
This approach is further improved by introducing a
layer of variational regularization, which addresses
some deficiencies of GNNs acting on sparse data.
The work is innovative for addressing insufficiencies
in state-of-the-art models. In prior work, the
Transformer model is said to have difficulty learning
attention weights from EHR data without guidance.
The Graph Convolution Transformer (GCT)
model demonstrated impressive performance on
EHR data, but required the input of pre-defined
medical ontologies. The work of this paper demonstrates
superior performance to both of these methods,
as well as classical methods like CNN and
RNN.

## Code Execution

The source code is adapted from the paper. It was provided by the authors but is
modified and extended for the purpose of reproducing the original research
published by (Zhu, W., & Razavian, N. 2021, April). The preprocess source code 
was provided by (Choi, E., Xu, Z., Li, Y., Dusenberry, M. W., Flores, G., Xue, Y., & Dai, A. M. 2019). 
This source code was modified to work with later TensorFlow APIs. 
Some additional source code is original for the purpose of reducing the complexity
of reproducing the research. This code will be available in a Jupyter Notebook.

The below examples do not represent all combinations.

### Dataset Preprocess :
```
python3 preprocess_eicu.py --input_path /buckets/ehr/eicu --output_path /buckets/ehr/eicu/out/
```

### MIMIC-III Training (No regularization):
```
python3 train.py --data_path /buckets/ehr/mimic/out/ --embedding_size 768 --dropout 0.2 --batch_size 10  --reg False --result_path /buckets/ehr/mimic/models/no_reg/
```

### MIMIC-III Training (With regularization):
```
python3 train.py --data_path /buckets/ehr/mimic/out/ --embedding_size 768 --dropout 0.2 --batch_size 10  --reg True --result_path /buckets/ehr/mimic/models/
```

### eICU Training (No regularization):
```
python3 train.py --data_path /buckets/ehr/eicu/out/ --embedding_size 128 --batch_size 32 --reg False  --result_path /buckets/ehr/eicu/models/batch32/reg_false
```

### eICU Training (With regularization):
```
python3 train.py --data_path /buckets/ehr/eicu/out/ --embedding_size 128 --batch_size 32 --reg True  --result_path /buckets/ehr/eicu/models/batch32/
```

### Config File Usage:
A configuration file may be passed as a command line argument. 

```
python3 train.py --config_path ./config.yaml
```
[Example Configuration File](./config.yaml)

## References

Zhu, W., & Razavian, N. (2021, April). Variationally regularized graph-based representation learning for electronic health records. In Proceedings of the Conference on Health, Inference, and Learning (pp. 1-13).

[Paper](https://doi.org/10.1145/3450439.3451855)

Choi, E., Xu, Z., Li, Y., Dusenberry, M. W., Flores, G., Xue, Y., & Dai, A. M. (2019). Graph convolutional transformer: Learning the graphical structure of electronic health records. arXiv preprint arXiv:1906.04716.

[Paper](https://arxiv.org/pdf/1906.04716.pdf)


## Demo Video

## Team

| Author           | Email                   | Contribution
|------------------|-------------------------|------------
| Sean Enright | seanre2@illinois.edu  | Coding, Documentation, Demo
| Charles Stolz    | cstolz2@illinois.edu    | Coding, Documentation, Demo
