"""
utility functions
"""
import os


def check_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Make {path}')
    else:
        print(f'{path} does exist')
