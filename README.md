# CS598: Deep Learning for Healthcare Reproducibility Project

## Project Overview

## Source code

Source code is adapted from the paper. It was provided by the authors. There
are also references in the source provided by the authors to other code they
adapted to preprocess the datasets. We credit the authors for the source code and
extend it for the purpose of reproducing the research.

```
python3 /home/rtikes/cs598-project/data/eicu/preprocess_eicu.py --input_path /media/rtikes/buckets/ehr/eicu --output_path /media/rtikes/buckets/ehr/eicu/out/

python3 /home/rtikes/cs598-project/train.py --data_path /media/rtikes/buckets/ehr/eicu/out/ --embedding_size 128 --result_path /media/rtikes/buckets/ehr/eicu/models/
```

## References

Zhu, W., & Razavian, N. (2021). Variationally regularized graph-based representation learning for electronic health records. ArXiv (Cornell University). [Paper](https://doi.org/10.1145/3450439.3451855)

## Demo Video

## Team

| Author           | Email                   | Contribution
|------------------|-------------------------|------------
| Sean Enright | seanre2@illinois.edu  | Coding, Documentation, Demo
| Charles Stolz    | cstolz2@illinois.edu    | Coding, Documentation, Demo
