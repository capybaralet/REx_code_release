import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch import nn, optim, autograd


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=41)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=str2bool, default=False)
parser.add_argument('--batch_size', type=int, default=25000)
parser.add_argument('--train_set_size', type=int, default=50000)
parser.add_argument('--eval_interval', type=int, default=100)
parser.add_argument('--print_eval_intervals', type=str2bool, default=True)

parser.add_argument('--train_env_1__color_noise', type=float, default=0.2)
parser.add_argument('--train_env_2__color_noise', type=float, default=0.1)
#parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument('--test_env__color_noise', type=float, default=0.9)

parser.add_argument('--erm_amount', type=float, default=1.0)

parser.add_argument('--early_loss_mean', type=str2bool, default=True)

parser.add_argument('--rex', type=str2bool, default=True)
parser.add_argument('--mse', type=str2bool, default=True)

parser.add_argument('--runs', type=int, default=5)

parser.add_argument('--plot', type=str2bool, default=True)
parser.add_argument('--save_numpy_log', type=str2bool, default=True)

flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

num_batches = (flags.train_set_size // 2) // flags.batch_size

# TODO: logging
all_train_nlls = -1*np.ones((flags.n_restarts, flags.steps))
all_train_accs = -1*np.ones((flags.n_restarts, flags.steps))
all_grayscale_test_accs = -1*np.ones((flags.n_restarts, flags.steps))
all_irmv1_penalties = -1*np.ones((flags.n_restarts, flags.steps))
all_rex_penalties = -1*np.ones((flags.n_restarts, flags.steps))

all_test_accs = -1*np.ones((flags.n_restarts, flags.runs, flags.steps))

wfs = (np.linspace(0, 400, flags.n_restarts))

final_train_accs = []
final_test_accs = []
highest_test_accs = []
for restart in range(flags.n_restarts):
  for run in range(flags.runs):
    print("Restart", restart, "run", run)
    penalty_anneal_iters_this_run = int(wfs[restart])
  
    highest_test_acc = 0.0

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e):
      def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()
      def torch_xor(a, b):
        return (a-b).abs() # Assumes both inputs are either 0 or 1
      # 2x subsample for computational convenience
      images = images.reshape((-1, 28, 28))[:, ::2, ::2]
      # Assign a binary label based on the digit; flip label with probability 0.25
      labels = (labels < 5).float()
      labels = torch_xor(labels, torch_bernoulli(.25, len(labels)))
      # Assign a color based on the label; flip the color with probability e
      colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
      # Apply the color to the image by zeroing out the other color channel
      images = torch.stack([images, images], dim=1)
      images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
      if use_cuda:
        return {
          'images': (images.float() / 255.).cuda(),
          'labels': labels[:, None].cuda()
        }
      else:
        return {
          'images': (images.float() / 255.),
          'labels': labels[:, None]
        }

    envs = [
      make_environment(mnist_train[0][::2], mnist_train[1][::2], flags.train_env_1__color_noise),
      make_environment(mnist_train[0][1::2], mnist_train[1][1::2], flags.train_env_2__color_noise),
      make_environment(mnist_val[0], mnist_val[1], flags.test_env__color_noise)
    ]

    # Define and instantiate the model

    class MLP(nn.Module):
      def __init__(self):
        super(MLP, self).__init__()
        if flags.grayscale_model:
          lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
          lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
      def forward(self, input):
        if flags.grayscale_model:
          out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out

    if use_cuda:
      mlp = MLP().cuda()
    else:
      mlp = MLP()

    # Define loss function helpers

    def mean_nll(logits, y):
      return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
      preds = (logits > 0.).float()
      return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
      if use_cuda:
        scale = torch.tensor(1.).cuda().requires_grad_()
      else:
        scale = torch.tensor(1.).requires_grad_()
      loss = mean_nll(logits * scale, y)
      grad = autograd.grad(loss, [scale], create_graph=True)[0]
      return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
      col_width = 13
      def format_val(v):
        if not isinstance(v, str):
          v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
      str_values = [format_val(v) for v in values]
      print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'test acc', 'irmv1 penalty', 'rex penalty')

    i = 0
    for step in range(flags.steps):
      n = i % num_batches
      for edx, env in enumerate(envs):
        if edx != len(envs) - 1:
          logits = mlp(env['images'][n*flags.batch_size:(n+1)*flags.batch_size])
          env['nll'] = mean_nll(logits, env['labels'][n*flags.batch_size:(n+1)*flags.batch_size])
          env['acc'] = mean_accuracy(logits, env['labels'][n*flags.batch_size:(n+1)*flags.batch_size])
          env['penalty'] = penalty(logits, env['labels'][n*flags.batch_size:(n+1)*flags.batch_size])
        else:
          logits = mlp(env['images'])
          env['nll'] = mean_nll(logits, env['labels'])
          env['acc'] = mean_accuracy(logits, env['labels'])
          env['penalty'] = penalty(logits, env['labels'])
      i+=1

      train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
      train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
      irmv1_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

      if use_cuda:
        weight_norm = torch.tensor(0.).cuda()
      else:
        weight_norm = torch.tensor(0.)
      for w in mlp.parameters():
        weight_norm += w.norm().pow(2)

      loss1 = envs[0]['nll']
      loss2 = envs[1]['nll']

      if flags.early_loss_mean:
        loss1 = loss1.mean()
        loss2 = loss2.mean()

      loss = 0.0
      loss += flags.erm_amount * (loss1 + loss2)

      loss += flags.l2_regularizer_weight * weight_norm

      penalty_weight = (flags.penalty_weight 
        if step >= penalty_anneal_iters_this_run else 1.0)
        #if step >= flags.penalty_anneal_iters else 1.0)

      if flags.mse:
        rex_penalty = (loss1.mean() - loss2.mean()) ** 2
      else:
        rex_penalty = (loss1.mean() - loss2.mean()).abs()

      if flags.rex:
        loss += penalty_weight * rex_penalty
      else:
        loss += penalty_weight * irmv1_penalty

      if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      test_acc = envs[2]['acc']

      if step % flags.eval_interval == 0:
        train_acc_scalar = train_acc.detach().cpu().numpy()
        test_acc_scalar = test_acc.detach().cpu().numpy()
        if flags.print_eval_intervals:
          pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy(),
            irmv1_penalty.detach().cpu().numpy(),
            rex_penalty.detach().cpu().numpy()
          )
        if (train_acc_scalar >= test_acc_scalar) and (test_acc_scalar > highest_test_acc):
          highest_test_acc = test_acc_scalar

      if flags.plot or flags.save_numpy_log:
        all_train_nlls[restart, step] = train_nll.detach().cpu().numpy()
        all_train_accs[restart, step] = train_acc.detach().cpu().numpy()
        all_irmv1_penalties[restart, step] = irmv1_penalty.detach().cpu().numpy()
        all_rex_penalties[restart, step] = rex_penalty.detach().cpu().numpy()
        #all_grayscale_test_accs[restart, step] = grayscale_test_acc.detach().cpu().numpy()

        all_test_accs[restart, run, step] = test_acc.detach().cpu().numpy()

  print('highest test acc this run:', highest_test_acc)

  final_train_accs.append(train_acc.detach().cpu().numpy())
  final_test_accs.append(test_acc.detach().cpu().numpy())
  highest_test_accs.append(highest_test_acc)
  print('Final train acc (mean/std across restarts so far):')
  print(np.mean(final_train_accs), np.std(final_train_accs))
  print('Final test acc (mean/std across restarts so far):')
  print(np.mean(final_test_accs), np.std(final_test_accs))
  print('Highest test acc (mean/std across restarts so far):')
  print(np.mean(highest_test_accs), np.std(highest_test_accs))

if flags.plot:
  plot_x = np.linspace(0, flags.steps, flags.steps)
  from pylab import *
  figure()
  title('test_acc vs iteration_that_penalty_is_activated_on')
  if flags.rex:
    title('test_acc vs iteration_that_rex_penalty_is_activated_on')
  else:
    title('test_acc vs iteration_that_irmv1_penalty_is_activated_on')

  all_test_accs_post_mean = all_test_accs.mean(1)
  plot(wfs[:flags.n_restarts], all_test_accs_post_mean[:, -1], label='test_acc')

  if flags.rex:
    xlabel('iteration_that_rex_penalty_is_activated_on')
  else:
    xlabel('iteration_that_irmv1_penalty_is_activated_on')
  ylabel('test_accuracy')
  legend(loc="upper right")
  savefig('train_acc__grayscale_test_acc__wf_x' + '_rex_' + str(flags.rex) + '.pdf')

if flags.save_numpy_log:
  import os
  directory = "np_arrays"
  if not os.path.exists(directory):
      os.makedirs(directory)

  outfile = "all_test_accs"
  np.save(directory + "/" + outfile + "_rex_" + str(flags.rex), all_test_accs)

  #outfile = "all_grayscale_test_accs"
  #np.save(directory + "/" + outfile, all_grayscale_test_accs)