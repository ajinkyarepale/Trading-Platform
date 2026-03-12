from setuptools import setup, find_packages

setup(
    name="quanttrader",
    version="1.0.0",
    description="Production-ready quantitative trading platform",
    author="QuantTrader",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    entry_points={
        "console_scripts": [
            "trade=cli:trade",
        ],
    },
    install_requires=open("requirements.txt").read().splitlines(),
)
