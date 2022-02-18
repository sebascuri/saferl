"""Environment Module.

The environments have n-dimensional returns, where the first component
is the return, and the following n-1 components are the constraints.

The constraints can be either indicator functions, where:
    - 0 means safe,
    - 1 means unsafe.

Or there are continuous functions, where:
    - <= 0 means safe,
    - >0 means unsafe.
"""
