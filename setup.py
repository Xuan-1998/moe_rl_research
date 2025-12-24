from setuptools import setup, find_packages

setup(
    name='moe_rl_research',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'stable-baselines3',
        'gymnasium',
        'numpy'
    ],
)
