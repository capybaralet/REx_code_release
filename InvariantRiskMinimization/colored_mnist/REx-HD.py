import argparse
import numpy as np
import torch
import copy

from torch import nn, optim, autograd
from torchvision import datasets


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description="Colored MNIST")
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--n_restarts", type=int, default=1)
parser.add_argument("--penalty_weight", type=float, default=1e4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--penalty_delay", type=int, default=100)
parser.add_argument("--grayscale_model", type=str2bool, default=False)
parser.add_argument("--train_set_size", type=int, default=50000)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--dropout", type=float, default=0.5)

parser.add_argument("--train_env_1__color_noise", type=float, default=0.2)
parser.add_argument("--train_env_2__color_noise", type=float, default=0.1)
# parser.add_argument('--val_env__color_noise', type=float, default=0.1)
parser.add_argument("--test_env__color_noise", type=float, default=0.9)

parser.add_argument("--wandb", type=str2bool, default=False)
#
parser.add_argument("--plot", type=int, default=1)
parser.add_argument("--plot_color", type=str, default=None)
parser.add_argument("--REx_HD", type=int, default=1)
parser.add_argument("--hetero", type=int, default=0) # if != 0, make 0s and 5s harder than other digit types
parser.add_argument("--digit_shift", type=int, default=0) # if != 0, shift probability of hard digits (0s/5s) across domains
#parser.add_argument("--extra_waterfall", type=int, default=1)

parser.add_argument("--irm", type=int, default=0)

config = parser.parse_args()

if config.wandb:
    import wandb
    wandb.init(project="new-rex-hp", config=config)


print("Config:")
for k, v in sorted(vars(config).items()):
    print("\t{}: {}".format(k, v))


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def irmv1_penalty(logits, y):
    if use_cuda:
        scale = torch.tensor(1.0).cuda().requires_grad_()
    else:
        scale = torch.tensor(1.0).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


class MLP(nn.Module):
    def __init__(self, dropout=0):
        super(MLP, self).__init__()
        if config.grayscale_model:
            lin1 = nn.Linear(14 * 14, config.hidden_dim)
        else:
            lin1 = nn.Linear(2 * 14 * 14, config.hidden_dim)
        lin2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        lin3 = nn.Linear(config.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        if dropout != 0:
            self._main = nn.Sequential(
                lin1,
                nn.ReLU(True),
                nn.Dropout(dropout),
                lin2,
                nn.Dropout(dropout),
                nn.ReLU(True),
                lin3,
            )
        else:
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3,)

    def forward(self, input):
        if config.grayscale_model:
            out = input.view(-1, 2, 14 * 14).sum(dim=1)
        else:
            out = input.view(-1, 2 * 14 * 14)
        out = self._main(out)
        return out


def trained_model_p_yx(xs, ys, batch_size=512, dropout=config.dropout):
    if use_cuda:
        model = MLP(dropout=dropout).cuda()
    else:
        model = MLP(dropout=dropout)

    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    n_tr = int(0.9 * len(xs))

    x_tr = xs[:n_tr]
    y_tr = ys[:n_tr]
    x_te = xs[n_tr:]
    y_te = ys[n_tr:]

    print("training model_p_yx")
    best_loss = np.inf
    for step in range(200):

        if True:
            batch_inds = np.random.randint(0, n_tr, batch_size)
            batch_x = x_tr[batch_inds]
            batch_y = y_tr[batch_inds]
        else:
            batch_x = x_tr
            batch_y = y_tr

        logits = model(batch_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            te_logits = model(x_te)
            te_loss = nn.functional.binary_cross_entropy_with_logits(te_logits, y_te)
            te_acc = ((te_logits > 0).float() == y_te).float().mean()
        # print(loss.detach().cpu().numpy(), te_loss.cpu().numpy(), te_acc.cpu().numpy())
        if te_loss < best_loss:
            best_loss = te_loss
            best_acc = te_acc
            best_step = step
            best_model = copy.deepcopy(model.state_dict())

    print(
        "training finished, best model at step %s (loss %s; acc %s)"
        % (best_step, best_loss.detach().cpu().numpy(), best_acc.cpu().numpy())
    )
    model.load_state_dict(best_model)
    return model


def make_environment(images, labels, e, grayscale_dup=False, hetero=False, p_hard=0.2):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    def torch_and(a, b):
        return a + b == 2  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # 0s and 5s are more likely to flip (i.e. harder to predict)
    is_hard = labels % 5 == 0
    hard_inds = torch.nonzero(is_hard)
    is_hard = (is_hard).float()
    n_hard = int(sum(is_hard))
    is_easy = labels % 5 != 0
    easy_inds = torch.nonzero(is_easy)
    is_easy = (is_easy).float()
    n_easy = int(sum(is_easy))
    print(images.shape, labels.shape, is_hard.shape)
    if p_hard != 0.2: 
        if p_hard < 0.2:  # remove some 0s/5s and replace them with other digits
            to_keep = torch_xor(
                1.0 - is_hard,
                torch_and(is_hard, torch_bernoulli(p_hard / 0.2, len(labels))).float(),
            )
            n_to_replace = int((1 - to_keep).sum())
            to_keep = torch.nonzero(to_keep)
            to_add = torch.randint(0, n_easy, (n_to_replace,))
            to_add = easy_inds[to_add]
        else:  # remove some other digits and replace them with 0s/5s
            to_keep = torch_xor(
                is_hard,
                torch_and(
                    1.0 - is_hard, torch_bernoulli((1 - p_hard) / 0.8, len(labels))
                ).float(),
            )
            n_to_replace = int((1 - to_keep).sum())
            to_keep = torch.nonzero(to_keep)
            to_add = torch.randint(0, n_hard, (n_to_replace,))
            to_add = hard_inds[to_add]
        # print(to_keep)
        # print(to_add)
        # to_keep = to_keep.bool()
        # to_add = to_add.bool()
        images = torch.cat((images[to_keep], images[to_add])).squeeze()
        labels = torch.cat((labels[to_keep], labels[to_add])).squeeze()
        is_hard = torch.cat((is_hard[to_keep], is_hard[to_add])).squeeze()
        print(images.shape, labels.shape, is_hard.shape)
    digits = labels
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    if hetero:
        labels = torch_xor(
            labels, torch_bernoulli(0.4, len(labels))
        #) * is_hard + torch_xor(labels, torch_bernoulli(0.2125, len(labels))) * (
        ) * is_hard + torch_xor(labels, torch_bernoulli(0.1, len(labels))) * (
            1 - is_hard
        )
    else:
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)

    if not grayscale_dup:
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

    if use_cuda:
        return {
            "images": (images.float() / 255.0).cuda(),
            "labels": labels[:, None].cuda(),
            "digits": digits,
            "is_hard": is_hard,
        }
    else:
        return {
            "images": (images.float() / 255.0),
            "labels": labels[:, None],
            "digits": digits,
            "is_hard": is_hard,
        }


def mean_accuracy(logits, y):
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


all_tr_accs = []
all_va_accs = []

for restart in range(config.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples
    num_train = config.train_set_size
    mnist = datasets.MNIST("~/datasets/mnist", train=True, download=True)
    mnist_train = (mnist.data[:num_train], mnist.targets[:num_train])
    mnist_val = (mnist.data[num_train:], mnist.targets[num_train:])

    ids = np.random.choice(num_train, size=num_train, replace=False)
    mnist_train = [d[ids] for d in mnist_train]

    if config.digit_shift:
        envs = [
            make_environment(
                mnist_train[0][::2],
                mnist_train[1][::2],
                config.train_env_1__color_noise,
                hetero=config.hetero,
                p_hard=0.3,
            ),
            make_environment(
                mnist_train[0][1::2],
                mnist_train[1][1::2],
                config.train_env_2__color_noise,
                hetero=config.hetero,
                p_hard=0.7,
            ),
            make_environment(
                mnist_val[0],
                mnist_val[1],
                config.test_env__color_noise,
                hetero=config.hetero,
                p_hard=0.5,
            ),
            make_environment(
                mnist_val[0],
                mnist_val[1],
                config.test_env__color_noise,
                grayscale_dup=True,
                hetero=config.hetero,
                p_hard=0.2,
            ),
        ]
    else:
        envs = [
            make_environment(
                mnist_train[0][::2],
                mnist_train[1][::2],
                config.train_env_1__color_noise,
                hetero=config.hetero,
            ),
            make_environment(
                mnist_train[0][1::2],
                mnist_train[1][1::2],
                config.train_env_2__color_noise,
                hetero=config.hetero,
            ),
            make_environment(
                mnist_val[0],
                mnist_val[1],
                config.test_env__color_noise,
                hetero=config.hetero,
            ),
            make_environment(
                mnist_val[0],
                mnist_val[1],
                config.test_env__color_noise,
                grayscale_dup=True,
                hetero=config.hetero,
            ),
        ]

    model = MLP().cuda() if use_cuda else MLP()

    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    #############################################################################
    # REx_HD uses P_data(x), P_model(y|x)

    tr_accs = []
    va_accs = []

    penalty_weight = 1.0

    n_train_domains = 2
    n_examples_per_domain = config.train_set_size // n_train_domains
    n_examples_per_domain = 25000
    batch_size_per_domain = n_examples_per_domain

    # lists of train data for each environment
    all_x = [env["images"][:n_examples_per_domain] for env in envs[:n_train_domains]]
    all_y = [env["labels"][:n_examples_per_domain] for env in envs[:n_train_domains]]
    va_x = envs[n_train_domains]["images"]
    va_y = envs[n_train_domains]["labels"]

    model_p_yx = [trained_model_p_yx(xs, ys) for xs, ys in zip(all_x, all_y)]

    # precompute all example weights (for use_model_p_x case)
    example_weights = []
    for d_x in range(n_train_domains):
        example_weights.append(
            torch.cat(
                [
                    torch.zeros(d_x * n_examples_per_domain),
                    torch.ones(n_examples_per_domain),
                    torch.zeros((n_train_domains - d_x - 1) * n_examples_per_domain),
                ]
            ).view(-1, 1)
        )
        if use_cuda:
            example_weights = [w.cuda() for w in example_weights]

    # precompute all predictions (for use_model_p_yx case)
    if config.REx_HD:
        with torch.no_grad():
            logits_yx = [m(torch.cat(all_x)) for m in model_p_yx]
            p_yx = [l.exp() / (1 + l.exp()) for l in logits_yx]
    else:
        p_yx = [torch.cat(all_y) for m in model_p_yx]

    for step in range(config.steps):
        # batch_x = all_x
        # batch_y = all_y

        # make predictions on the combined data:
        logits = model(torch.cat(all_x))
        per_example_losses = [
            nn.functional.binary_cross_entropy_with_logits(logits, p, reduction="none")
            for p in p_yx
        ]

        train_acc = torch.stack(
            [mean_accuracy(model(x), y) for x, y in zip(all_x, all_y)]
        ).mean()

        irmv1_penalties = torch.stack([irmv1_penalty(logits, p) for p in p_yx]).mean()
        # breakpoint()

        if config.REx_HD:
            # compute all hybrid-domain risks, inds: [P(x), P(y|x)]
            total_risk = 0
            # also compute all penalty terms, inds: [P(x)]
            total_penalty = 0
            for d_x in range(n_train_domains):  # for all i
                all_risks_i = []
                for d_yx in range(n_train_domains):  # for all j,k
                    # when use_model_p_x, example_weights depends on the density model (indexed by d_x);
                    # else, example_weights are just 0/1 indicators of whether a given example belongs to domain d_x;
                    # when use_model_p_yx, per_example_losses depends on the predictor (indexed by d_yx);
                    # else, per_example_losses are given by the true labels, but we only use examples from domain d_yx (this is also accomplished via 0/1 masking)
                    this_risk = (example_weights[d_x] * per_example_losses[d_yx]).mean()
                    all_risks_i.append(this_risk)
                    total_risk += this_risk
                #assert n_train_domains == 2
                this_penalty = (torch.stack(all_risks_i)).var()
                total_penalty += this_penalty
        else:
            # compute all risks (same domain for P(X), P(Y|X)), inds: [Domain]
            total_risk = 0
            all_risks = []
            for d_x in range(n_train_domains):  # for all i
                this_risk = (example_weights[d_x] * per_example_losses[d_x]).mean()
                all_risks.append(this_risk)
                total_risk += this_risk
            total_penalty = torch.var(torch.stack(all_risks))

        # import ipdb; ipdb.set_trace()
        if config.irm:
            loss = total_risk / penalty_weight + irmv1_penalties
        else:
            loss = total_risk / penalty_weight + total_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = {
            "loss": loss,
            "total_penalty": total_penalty,
            "total_risk": total_risk.detach().cpu().numpy(),
            "valid_acc": mean_accuracy(model(va_x), va_y).detach().cpu().numpy(),
            "train_acc": train_acc.detach().cpu().numpy(),
        }
        print("train acc: %.4f\t valid acc: %.4f" % (log["train_acc"], log["valid_acc"]))
        tr_accs.append(log["train_acc"])
        va_accs.append(log["valid_acc"])

        if config.wandb:
            wandb.log(log)

        if step >= config.penalty_delay:
            print("---")
           # penalty_weight = 1.05**(step-config.penalty_delay) * config.penalty_weight
            penalty_weight = config.penalty_weight
    all_tr_accs.append(tr_accs[-1])
    all_va_accs.append(va_accs[-1])

    print(np.mean(all_tr_accs), np.std(all_tr_accs))
    print(np.mean(all_va_accs), np.std(all_va_accs))

    #tr_higher = [tr > va for tr,va in zip(tr_accs, va_accs)]
    filtered_va = [-1 if tr < va else va for tr,va in zip(tr_accs, va_accs)]
    best_step = np.argmax(filtered_va)
    print(tr_accs[best_step], va_accs[best_step])

    if config.plot:
        summ = sum
        from pylab import *#plot, figure, show, title
        sum = summ
        figure()
        #title('REx_HD={}'.format(config.REx_HD))
        ylabel('accuracy')
        xlabel('epoch')
        plot(va_accs, label='valid')
        plot(tr_accs, '--', label='train')
        legend()


