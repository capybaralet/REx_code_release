
import argparse
import numpy as np
import numpy.lib as npl
import torch
from torch import nn, optim, autograd

from load import get_years, get_sectors, YEARS
from utils import *


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--irm_penalty_weight', type=float, default=10000.0)
parser.add_argument('--rex_penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save', type=str, default='')
parser.add_argument('--train_envs', type=str, default='2014,2015,2016')
parser.add_argument('--test_envs', type=str, default='')
flags = parser.parse_args()

train_env_ids = [int(s.strip()) for s in flags.train_envs.split(',')]
if flags.test_envs:
    test_env_ids = [int(s.strip()) for s in flags.test_envs.split(',')]
else:
    test_env_ids = npl.setxor1d(YEARS, train_env_ids)


print('Flags:')
for k,v in sorted(vars(flags).items()):
    print("    {}: {}".format(k, v))


def whiten(x):
    with torch.no_grad():
        x -= x.mean(dim=0)
        x /= x.std(dim=0)
    return x

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def env_irm_penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.mean(grad**2)

def get_rex_penalty(train_envs):
    losses = torch.stack([e['nll'] for e in train_envs])
    penalty = torch.var(losses)
    return penalty


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        lin1 = nn.Linear(input_size, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            lin1, nn.Tanh(), #nn.ReLU(True),
            nn.Dropout(),
            lin2, nn.Tanh(), #nn.ReLU(True),
            nn.Dropout(),
            lin3)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_size)
        out = self._main(x)
        return out


final_train_accs = []
final_test_accs = []
logs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    train_envs = get_years(train_env_ids)
    test_envs = get_years(test_env_ids)
    # preprocess
    for e in train_envs + test_envs:
        e['images'] = whiten(e['images'])
    print_env_info(train_envs, test_envs)

    # init
    logger = Logger()
    mlp = MLP(train_envs[0]['images'].shape[1]).cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'irm penalty', 'rex penalty', 'test acc')

    for step in range(flags.steps):
        for env in train_envs + test_envs:
            env['logits'] = mlp(env['images'])
            env['nll'] = mean_nll(env['logits'], env['labels'])
            env['acc'] = mean_accuracy(env['logits'], env['labels'])
            env['penalty'] = env_irm_penalty(env['logits'], env['labels'])

        train_nll = torch.stack([e['nll'] for e in train_envs]).mean()
        train_acc = torch.stack([e['acc'] for e in train_envs]).mean()
        irm_penalty = torch.stack([e['penalty'] for e in train_envs]).mean()
        rex_penalty = get_rex_penalty(train_envs)

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        if flags.irm_penalty_weight:
            if step >= flags.penalty_anneal_iters:
                loss /= flags.irm_penalty_weight
            loss += irm_penalty

        elif flags.rex_penalty_weight:
            if step >= flags.penalty_anneal_iters:
                loss /= flags.rex_penalty_weight
            loss += rex_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log('train_nll', train_nll)
        logger.log('train_acc', train_acc)
        logger.log('irm_penalty', irm_penalty)
        logger.log('rex_penalty', rex_penalty)
        logger.log('test_acc', [e['acc'] for e in test_envs])
        logger.log('losses', [e['nll'] for e in train_envs])

        if step % 100 == 0:
            print_stats(step, logger)

    final_train_accs.append(np.mean(logger['train_acc'][-50:]))
    final_test_accs.append(np.mean(logger['test_acc'][-50:]))
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))

    logs.append(logger)

    if flags.plot:
        plot(logger)

if flags.save:
    save(logs, 'results/%s_%s_%s' % (flags.save, ','.join([str(e) for e in train_env_ids]), ','.join([str(e) for e in test_env_ids])))

