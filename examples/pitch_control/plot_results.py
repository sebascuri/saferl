"""Python Script Template."""
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from examples.utilities import set_figure_params


RESULTS = OrderedDict(
    SafetyFilter=[0.9, 0.1, 0.1], SafeMPC=[0.1, 0.1, 0.1], CMDP=[1.0, 0.1, 1.0],
)

set_figure_params()
fig, ax = plt.subplots(ncols=1, nrows=1, sharex="all")
labels = [r"\textbf{SafetyFilter}", "SafeMPC", "CMDP"]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

ax.bar(
    x - width,
    [result[0] for result in RESULTS.values()],
    width,
    color="C0",
    label="J_r",
)
ax.bar(x, [result[1] for result in RESULTS.values()], width, color="C1", label="J_c")
ax.bar(
    x + width,
    [result[2] for result in RESULTS.values()],
    width,
    color="C2",
    label="ConstraintViolation",
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, fontsize=8)
plt.legend()
plt.show()
