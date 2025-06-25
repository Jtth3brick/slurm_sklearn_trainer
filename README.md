Uses Redis from Docker Image to manage argument queue and results to delegate arguments and overcome slow disk i/o

Designed to allow as many workers as needed to help search

Make sure to make logs dir for slurm output before starting the manager

Replace the get_data method in manager to change your data input

Hyperparam search is stored in `config.yaml` among other helpful items.

Info on model cacheing

make sure to manually create logs dir
