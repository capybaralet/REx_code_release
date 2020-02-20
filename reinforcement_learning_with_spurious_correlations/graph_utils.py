import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
import csv
import math
from collections import defaultdict


def read_log_file(file_name, key_name, value_name, smooth=3):
    keys, values = [], []
    try:
        with open(file_name, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                try:
                    key, value = line[key_name], line[value_name]
                    keys.append(int(key))
                    values.append(float(value))
                except:
                    pass
    except:
        print('bad file: %s' % file_name)
        return None, None
    keys, values = np.array(keys), np.array(values)
    if smooth > 1 and values.shape[0] > 0:
        K = np.ones(smooth)
        ones = np.ones(values.shape[0])
        values = np.convolve(values, K, 'same') / np.convolve(ones, K, 'same')
    return keys, values


def parse_log_files(
    file_name_template,
    key_name,
    value_names,
    num_seeds,
    best_k=None,
    max_key=False
):
    all_values = defaultdict(list)
    for value_name in value_names:
        all_keys = []
        actual_keys = None
        for seed in range(1, num_seeds + 1):
            file_name = file_name_template % seed
            keys, values = read_log_file(file_name, key_name, value_name)
            if keys is None or keys.shape[0] == 0:
                continue
            all_keys.append(keys)
            all_values[value_name].append(values)

    if len(all_values[value_name]) == 0:
        return None, None, None

    all_keys = sorted(all_keys, key=lambda x: x.shape[0])
    threshold = all_keys[-1].shape[0] if max_key else all_keys[0].shape[0]

    means, half_stds = defaultdict(list), defaultdict(list)
    for value_name in value_names:
        for i in range(threshold):
            vals = []

            for v in all_values[value_name]:
                if i < v.shape[0]:
                    vals.append(v[i])
            if best_k is not None:
                vals = sorted(vals)[-best_k:]
            means[value_name].append(np.mean(vals))
            half_stds[value_name].append(0.5 * np.std(vals) / math.sqrt(num_seeds))
        means[value_name] = np.array(means[value_name])
        half_stds[value_name] = np.array(half_stds[value_name])

        keys = all_keys[-1][:threshold]
        assert means[value_name].shape[0] == keys.shape[0]
    return keys, means, half_stds


def print_result(
    root,
    title,
    label=None,
    num_seeds=1,
    train=False,
    key_name='step',
    value_names=['episode_reward'],
    max_time=None,
    best_k=None,
    timescale=1,
    max_key=False
):
    linestyles = ['-', '-.', '--', ':']
    file_name = 'train.csv' if train else 'eval.csv'
    file_name_template = os.path.join(root, 'seed_%d', file_name)
    keys, means, half_stds = parse_log_files(
        file_name_template,
        key_name,
        value_names,
        num_seeds,
        best_k=best_k,
        max_key=max_key
    )
    label = label or root.split('/')[-1]
    if keys is None:
        return

    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    keys *= timescale

    color = None
    for idx, value_name in enumerate(value_names):
        if color is None:
            p = plt.plot(keys, means[value_name], linestyles[idx], label=label)
            color = p[0].get_color()
        else:
            p = plt.plot(keys, means[value_name], linestyle=linestyles[idx], color=color)
        plt.fill_between(keys, means[value_name] - half_stds[value_name], means[value_name] + half_stds[value_name], color=color, alpha=0.2)

    plt.locator_params(nbins=10, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    #plt.ylim(0, 1050)
    plt.tight_layout()

    plt.grid(alpha=0.8)
    plt.title(title)
    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')
    plt.xlabel(key_name)
    plt.ylabel('Episode Reward')


def plot_seeds(
    task,
    exp_query,
    root,
    train=True,
    key_name='step',
    value_name='episode_reward',
    num_seeds=10
):
    root = os.path.join(root, task)
    experiment = None
    for exp in os.listdir(root):
        if re.match(exp_query, exp):
            experiment = os.path.join(root, exp)
            break
    if experiment is None:
        return
    file_name = 'train.log' if train else 'eval.log'
    file_name_template = os.path.join(experiment, 'seed_%d', file_name)

    plt.locator_params(nbins=10, axis='x')
    plt.locator_params(nbins=10, axis='y')
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.title(task)

    plt.xlabel(key_name)
    plt.ylabel(value_name)

    for seed in range(1, num_seeds + 1):
        file_name = file_name_template % seed
        keys, values = read_log_file(file_name, key_name, value_name)
        if keys is None or keys.shape[0] == 0:
            continue

        plt.plot(keys, values, label='seed_%d' % seed, linewidth=0.5)

    plt.legend(loc='lower right', prop={
        'size': 6
    }).get_frame().set_edgecolor('0.1')

def plot_experiment(
    task,
    exp_query,
    root='runs',
    exp_ids=None,
    train=False,
    key_name='step',
    value_names=['eval_episode_reward'],
    neg_exp=None,
    num_seeds=10,
    max_time=None,
    best_k=None,
    timescale=1,
    max_key=False
):
    root = os.path.join(root, task)

    experiments = set()
    for exp in os.listdir(root):
        if re.match(exp_query, exp):
            if neg_exp is None or neg_exp not in exp:
                exp = os.path.join(root, exp)
                experiments.add(exp)

    exp_ids = list(range(len(experiments))) if exp_ids is None else exp_ids
    for exp_id, exp in enumerate(sorted(experiments)):
        if exp_id in exp_ids:
            print_result(
                exp,
                task,
                num_seeds=num_seeds,
                train=train,
                key_name=key_name,
                value_names=value_names,
                max_time=max_time,
                best_k=best_k,
                timescale=timescale,
                max_key=max_key
            )
