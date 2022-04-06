'''
Date: 2021-03-22 15:11:55
LastEditors: Chenhuiyu
LastEditTime: 2021-03-29 01:11:47
FilePath: \\03-28-SleepZzNet\\run_train.py
'''
import argparse
import os

from train import train


def run_train(dataset_name, output_dir, K_fold, random_seed):
    """[Start Training]c

    Args:
        dataset_name (str): dataset name, such as sleepedf/sleepedfx
        output_dir (str): path to save training logs,checkpoint,and predicted pictures
        K_fold (int): K-fold cross validation
        random_seed (int): random seed to select subjects file for cross validation
    """
    # dataset_name = 'sleepedf'
    # output_dir = '\\5_epochs_filter_theta'
    # random_seed = 0
    config_file = os.path.join('config', f'{dataset_name}.py')
    output_dir = os.path.join(output_dir, f'out_{dataset_name}_resnet')

    # todo: add cross-validation
    train(
        config_file=config_file,
        output_dir=output_dir,
        random_seed=random_seed,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='sleepedf')
    parser.add_argument('--output_dir', type=str, default='\\10_epochs')
    parser.add_argument('--K_fold', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=9)
    args = parser.parse_args()

    run_train(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        K_fold=args.K_fold,
        random_seed=args.random_seed,
    )
