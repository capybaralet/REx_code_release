import torch
import numpy as np
import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.data = dict()

    def log(self, k, v):
        if not k in self.data:
            self.data[k] = []
        v = self._to_np(v)
        self.data[k].append(v)

    def __getitem__(self, k):
        return self.data.get(k, [])

    def _to_np(self, v):
        if isinstance(v, torch.Tensor):
            with torch.no_grad():
                return v.cpu().numpy()
        if isinstance(v, list):
            return [self._to_np(v_) for v_ in v]
        return v

def plot(log):
    plt.figure()
    plt.semilogy(log['irm_penalty'])
    plt.semilogy(log['rex_penalty'])
    plt.legend(['irm penalty', 'rex penalty'])
    plt.figure()
    plt.plot(log['train_acc'], 'k')
    plt.plot(log['test_acc'], alpha=0.7)
    plt.legend(['train acc'] + ['test acc'] * len(log['test_acc'][0]))
    plt.figure()
    plt.plot(log['losses'], 'k', alpha=0.3)
    plt.title('losses')
    plt.ylabel('nll')
    plt.show()

def save(logs, name):
    data = dict()
    for k in logs[0].data.keys():
        data[k] = np.array([l[k] for l in logs])
        np.savetxt('%s_%s_mean.dat' % (name, k), data[k].mean(0))
        np.savetxt('%s_%s_std.dat' % (name, k), data[k].std(0))
    np.savez_compressed('%s.npz' % name, **data)

def print_stats(step, log):
    pretty_print(
        np.int32(step),
        np.mean(log['train_nll'][-50:]),
        np.mean(log['train_acc'][-50:]),
        np.mean(log['irm_penalty'][-50:]),
        np.mean(log['rex_penalty'][-50:]),
        *np.array(log['test_acc'][-50:]).mean(axis=0),
    )

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

def print_env_info(train_envs, test_envs):
    num_feat = train_envs[0]['images'].shape[1]
    print('training on', len(train_envs), 'environments (using', num_feat, 'features):')
    for e in train_envs:
        print('   ', e['info'], len(e['labels']))
    print('testing on:')
    for e in test_envs:
        print('   ', e['info'], len(e['labels']))

