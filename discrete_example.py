"""
This is a very simple setting to show the utility of REx-HD.

We have discrete x in (0,1,2) and y in (0,1).
x is represented as a one-hot.

P(y=1|x=0) = .5
P(y=1|x=1) = .75
P(y=1|x=2) = varies

We also have covariate shift.
p_x2 = .5
p_x0_given_x0or1 = varies

------------------------------
We learn parameters for each of the 3 conditional probabilities.
The stable solution (for BCE loss) is:
       (.5, .75, .5)
We expect the following solutions:
ERM    (.5, .75, mean(P(y=1|x=2)))
IRM    (.5, .75, .5)
REx    (.5, .5,  .5)
REx-HD (.5, .75, .5)

------------------------------
Thoughts:
I don't think there's any reason we need to vary p_x2, but we should try anyways...

How many environments do we need?
I guess we need 3... the model needs to see both directions of variation
maybe that's why the HCMNIST thing wasn't working!?

"""

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import grad


n_steps = 500
# lr_decay = .99


# TODO: plot these things somehow?
# REx would span a 2D subspace of a 4D space here...
# if we look at change in P(X) / P(Y|X) independently, then we can plot these as 1D subspaces of 2D spaces (i.e. the probability "plane")

env_p_yxs = [
    [0.5, 0.75, 0.1],
    [0.5, 0.75, 0.1],
    [0.5, 0.75, 0.2],
]
#env_p_xs = [
#    [0.4, 0.1, 0.5],
#    [0.35, 0.15, 0.5],
#    [0.25, 0.25, 0.5],
#]
env_p_xs = [
    [0.4, 0.1, 0.5],
    [0.1, 0.4, 0.5],
    [0.1, 0.4, 0.5],
]
#env_p_xs = [
#    [0.1, 0.4, 0.5],
#    [0.1, 0.4, 0.5],
#    [0.1, 0.4, 0.5],
#]  # NO covariate shift!  REx and REx-HD should be the same!?

env_p_xs = torch.tensor(env_p_xs)
env_p_yxs = torch.tensor(env_p_yxs)
x = torch.eye(3)


def bce_xy(model, env_x, env_y, w=1):
    ''' compute the loss for a given pair of envs for p_x and p_yx '''
    logits = model(x)
    p_x = env_p_xs[env_x].view(-1, 1)
    p_yx = env_p_yxs[env_y].view(-1, 1)
    bce = torch.binary_cross_entropy_with_logits(w * logits, p_yx)
    return p_x.T.mm(bce)

def rex_loss(model, penalty_weight):
    losses = [bce_xy(model, env, env) for env in range(3)]
    losses = torch.stack(losses)
    return losses.sum() / penalty_weight + penalty_weight * losses.var()

def erm_loss(model, penalty_weight):
    losses = [bce_xy(model, env, env) for env in range(3)]
    return torch.stack(losses).sum()

def rex_hd_loss(model, penalty_weight):
    losses = []
    for env_x in range(3):
        env_x_losses = [bce_xy(model, env_x, env_y) for env_y in range(3)]
        losses.append(torch.stack(env_x_losses))
    losses = torch.stack(losses)
    loss = 10 * losses.sum() / penalty_weight + penalty_weight * losses.var(dim=1).sum()
    return loss

def irm_loss(model, penalty_weight):
    w = torch.ones((3,1), requires_grad=True)
    losses = [bce_xy(model, env, env, w) for env in range(3)]
    losses = torch.stack(losses)
    penalty = grad(losses.sum(), w, create_graph=True)[0].pow(2).mean()
    return losses.sum() + min(penalty_weight, 100) * penalty


loss_fcts = {
    "REx-HD": rex_hd_loss,
    "REx": rex_loss,
    "IRM": irm_loss,
    "ERM": erm_loss,
    }


if __name__ == "__main__":

    plt.style.use('ggplot')

    plt.figure(figsize=(8.5,3))

    run_methods = ["REx-HD", "REx", "IRM", "ERM"]

    for method in run_methods:

        model = nn.Linear(3, 1, bias=False)
        model.weight.data.fill_(0)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
        loss_fct = loss_fcts[method]
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
        #loss_fct = methods[method]
        all_probs = []

        for step in range(n_steps):

            loss = loss_fct(model, penalty_weight=step+1)
            #loss = loss_fct(model, penalty_weight=1000)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logits = model.weight.detach()
            probs = logits.exp() / (1 + logits.exp())
            probs = probs.squeeze().numpy()
            all_probs.append(probs)
            print(probs)

        all_probs = np.array(all_probs)

        plt.subplot(131)
        plt.plot(all_probs[:, 0])
        plt.subplot(132)
        plt.plot(all_probs[:, 1])
        plt.subplot(133)
        plt.plot(all_probs[:, 2])


    plt.subplot(131)
    plt.ylim(0.25, 0.75)
    plt.title('$P(Y = 1 | X = 0)$')
    plt.ylabel('p')
    plt.subplot(132)
    plt.title('$P(Y = 1 | X = 1)$')
    plt.xlabel('Steps')
    plt.subplot(133)
    plt.title('$P(Y = 1 | X = 2)$')
    plt.legend(run_methods, facecolor="white")
    plt.tight_layout()

    plt.show()
