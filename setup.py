"""Setup script of saferl."""
from setuptools import find_packages, setup

extras = {
    "test": [
        "pytest>=5.0,<5.1",
        "flake8>=3.7.8,<3.8",
        "pydocstyle==4.0.0",
        "black>=19.10b0",
        "isort>=5.0.0",
        "pytest_cov>=2.7,<3",
        "mypy==0.750",
    ],
    "envs": [
        "box2d-py>=2.3.5",
        "atari_py>=0.2.6",
        "MinAtar @ git+ssh://git@github.com/kenjyoung/MinAtar@master#egg=MinAtar",
        "seaborn>=0.9.0",
        # "gym-minigrid>=1.0",
    ],
}
extras["all"] = [item for group in extras.values() for item in group]

setup(
    name="saferl",
    version="0.0.1",
    author="Sebastian Curi",
    author_email="sebascuri@gmail.com",
    license="MIT",
    python_requires=">=3.7.0",
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "rllib @ git+ssh://git@github.com/sebascuri/rllib@dev#egg=rllib",
        "hucrl @ git+ssh://git@github.com/sebascuri/hucrl@dev#egg=hucrl",
        # "safety_gym @ git+ssh://git@github.com/openai/safety-gym#egg=safety_gym"
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
