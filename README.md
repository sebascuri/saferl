# Implementation of Safety Filter Hallucinated RL Algorithm


To install create a conda environment:
```bash
$ conda create -n saferl python=3.7
$ conda activate saferl
```

```bash
$ pip install -e .[test]
```

The environments should define rewards and costs in such a way that
the objective is to maximize the sum of discounted rewards, subject
to the constraint that costs < 0. 