---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: float64
  - name: url
    dtype: string
  splits:
  - name: train
    num_bytes: 32699
    num_examples: 83
  download_size: 19141
  dataset_size: 32699
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for AIMO Validation AMC

All 83 come from AMC12 2022, AMC12 2023, and have been extracted from the AOPS wiki page https://artofproblemsolving.com/wiki/index.php/AMC_12_Problems_and_Solutions

This dataset serves as an internal validation set during our participation in the AIMO progress prize competition. Using data after 2021 is to avoid potential overlap with the MATH training set.

Here are the different columns in the dataset:
problem: the *modified* problem statement
answer: the adapted integer answer
url: url to the problem page in the website

## Dataset creation process
The original AMC12 problems are MCQ with 4 choices. In order to be closer to the AIMO progress prize condition, we modified the problem statement to have an integer output. Those problems whose statement can not be modified are rejected.

Example:

### Original problem:
```
Flora the frog starts at 0 on the number line and makes a sequence of jumps to the right. In any one jump, independent of previous jumps, Flora leaps a positive integer distance $m$ with probability $\frac{1}{2^m}$.
What is the probability that Flora will eventually land at 10?
$\textbf{(A)}~\frac{5}{512}\qquad\textbf{(B)}~\frac{45}{1024}\qquad\textbf{(C)}~\frac{127}{1024}\qquad\textbf{(D)}~\frac{511}{1024}\qquad\textbf{(E)}~\frac{1}{2}$
```

### Modified problem:
```
Flora the frog starts at 0 on the number line and makes a sequence of jumps to the right. In any one jump, independent of previous jumps, Flora leaps a positive integer distance $m$ with probability $\frac{1}{2^m}$.
What is the probability that Flora will eventually land at 10? Write the answer as a simplified fraction $\frac{m}{n}$, find $m+n$
```