"""Experiment utilities."""
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
from matplotlib import rcParams


@dataclass
class Experiment:
    """Definition of Experiment dataclass."""

    agent: str = field(default="cmdp",)
    hallucinate: bool = field(default=False)
    num_episodes: int = field(default=100)


parser = ArgumentParser(Experiment)


def set_figure_params(serif=False, fontsize=9):
    """Define default values for font, fontsize and use latex.

    Parameters
    ----------
    serif: bool, optional
        Whether to use a serif or sans-serif font.
    fontsize: int, optional
        Size to use for the font size

    """
    params = {
        # "font.serif": [
        #     "Times",
        #     "Palatino",
        #     "New Century Schoolbook",
        #     "Bookman",
        #     "Computer Modern Roman",
        # ]
        # + rcParams["font.serif"],
        # "font.sans-serif": [
        #     "Times",
        #     "Helvetica",
        #     "Avant Garde",
        #     "Computer Modern Sans serif",
        # ]
        # + rcParams["font.sans-serif"],
        # "font.family": "sans-serif",
        # "text.usetex": True,
        # Make sure mathcal doesn't use the Times style
        # "text.latex.preamble": r"\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}",
        "axes.labelsize": fontsize,
        "axes.linewidth": 0.75,
        "font.size": fontsize,
        "legend.fontsize": fontsize * 0.7,
        "xtick.labelsize": fontsize * 8 / 9,
        "ytick.labelsize": fontsize * 8 / 9,
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "legend.numpoints": 1,
    }

    if serif:
        params["font.family"] = "serif"

    rcParams.update(params)
