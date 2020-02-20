# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import math
import pdb
from sklearn.linear_model import LinearRegression
from itertools import chain, combinations
from scipy.stats import f as fdist
from scipy.stats import ttest_ind
import torch.nn.functional as F

from torch.autograd import grad

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

def compute_softmax_negentropy(loss1, loss2):
    ps = F.softmax(torch.stack([loss1, loss2]))
    return (ps[0] * torch.log(ps[0]) + ps[1] * torch.log(ps[1]))

class REXv21(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [1, 10, 100, 1000, 10000, 100000]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("REXv21 (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0, use_cuda=False):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.ones(dim_x,1))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        #opt = torch.optim.SGD([self.phi], lr=0.01, momentum=0.9) #args["lr"])
        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        MSE = torch.nn.MSELoss()
        bound = 0.5
        loss_weighting = .5 * torch.ones([2])
        lw_lr = 0.1
         
        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
                       
            x_e_1, y_e_1 = environments[0]
            x_e_2, y_e_2 = environments[1]
            r1 = MSE(x_e_1 @ self.phi, y_e_1)
            r2 = MSE(x_e_2 @ self.phi, y_e_2)

            if iteration < 1:
              bounds = torch.tensor([bound, 1-bound])
            elif r1 > r2:
              bounds = torch.tensor([bound, 1-bound]) 
            else:
              bounds = torch.tensor([1-bound, bound])

            if use_cuda:
              bounds = bounds.cuda()
            loss_weighting = (1 - lw_lr) * loss_weighting + lw_lr * bounds
            if iteration > 10000:
                beta = reg
            else:
                beta = 1.0
            loss = 0.0
            loss += beta * (compute_softmax_negentropy(r1, r2))
            loss += r1 + r2
            loss += 1e-4*self.phi.pow(2).sum().sqrt()
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                w_norm = self.phi.pow(2).sum().sqrt()
                print("it: {:05d} | beta: {:.5f} | loss: {:.5f} | r1: {:.5f} | r2: {:.5f} | w_norm: {:.5f} | {}".format(iteration,
                                                                        beta,
                                                                        loss,
                                                                        r1,
                                                                        r2,
                                                                        w_norm,
                                                                        w_str))
    def solution(self):
        return self.phi





class REXv1(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [1, 10, 100, 1000]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("REXv1 (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        # opt = torch.optim.SGD([self.phi], lr=0.01, momentum=0.9) #args["lr"])
        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            
            x_e_1, y_e_1 = environments[0]
            x_e_2, y_e_2 = environments[1]
            r1 = loss(x_e_1 @ self.phi, y_e_1)
            r2 = loss(x_e_2 @ self.phi, y_e_2)
            beta = reg
            r = 0.0
            if r1 > r2:
              r = r1 * (beta+1) - r2 * beta
            else:
              r = r2 * (beta+1) - r1 * beta
            # r += 1e-4*self.phi.pow(2).sum().sqrt()
            opt.zero_grad()
            r.backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                w_norm = self.phi.pow(2).sum().sqrt()
                print("{:05d} | reg: {:.5f} | error: {:.5f} | penalty: {:.5f} | weight_norm: {:.5f} | {}".format(iteration,
                                                                        reg,
                                                                        r,
                                                                        1e-4*self.phi.pow(2).sum().sqrt(),
                                                                        w_norm,
                                                                        w_str))

    def solution(self):
        return self.phi @ self.w



class REX_et(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            x_e_1, y_e_1 = environments[0]
            x_e_2, y_e_2 = environments[1]
            loss1 = loss(x_e_1 @ self.phi, y_e_1)
            loss2 = loss(x_e_2 @ self.phi, y_e_2)

            bound_init = 0.5
            bound = bound_init
            if iteration < -1:
              bounds = torch.tensor([bound_init, 1-bound_init])
            elif loss1 > loss2:
              bounds = torch.tensor([bound, 1-bound]) 
            else:
              bounds = torch.tensor([1-bound, bound])
            lw_lr = 0.5

            loss_weighting = .1 * torch.ones([2])
            loss_weighting = (1 - lw_lr) * loss_weighting + lw_lr * bounds 

            l = 0.0
            l += (loss_weighting[0] * loss1).mean() + (loss_weighting[1] * loss2 ).mean()

            opt.zero_grad()
            l.backward()
            opt.step()

            if args["verbose"] and iteration % 1000 == 0:
                w_str = pretty(self.solution())
                w_norm = self.phi.pow(2).sum().sqrt()
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                        reg,
                                                                        error,
                                                                        penalty,
                                                                        w_norm,
                                                                        w_str))

    def solution(self):
        return self.phi @ self.w


class IRMstyleERM(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM style ERM (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi, y_e)
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                w_norm = self.phi.pow(2).sum().sqrt()
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                        reg,
                                                                        error,
                                                                        penalty,
                                                                        w_norm,
                                                                        w_str))

    def solution(self):
        return self.phi @ self.w




class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if args["verbose"]:
                print("IRM (reg={:.3f}) has {:.3f} validation error.".format(
                    reg, err))

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()
        self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args["lr"])
        loss = torch.nn.MSELoss()

        for iteration in range(args["n_iterations"]):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()

            if args["verbose"] and iteration % 100 == 0:
                w_str = pretty(self.solution())
                w_norm = self.phi.pow(2).sum().sqrt()
                print("{:05d} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {}".format(iteration,
                                                                        reg,
                                                                        error,
                                                                        penalty,
                                                                        w_norm,
                                                                        w_str))

    def solution(self):
        return self.phi @ self.w


class InvariantCausalPrediction(object):
    def __init__(self, environments, args):
        self.coefficients = None
        self.alpha = args["alpha"]

        x_all = []
        y_all = []
        e_all = []

        for e, (x, y) in enumerate(environments):
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            e_all.append(np.full(x.shape[0], e))

        x_all = np.vstack(x_all)
        y_all = np.vstack(y_all)
        e_all = np.hstack(e_all)

        dim = x_all.shape[1]

        accepted_subsets = []
        for subset in self.powerset(range(dim)):
            if len(subset) == 0:
                continue

            x_s = x_all[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)

            p_values = []
            for e in range(len(environments)):
                e_in = np.where(e_all == e)[0]
                e_out = np.where(e_all != e)[0]

                res_in = (y_all[e_in] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y_all[e_out] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))

            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self.alpha:
                accepted_subsets.append(set(subset))
                if args["verbose"]:
                    print("Accepted subset:", subset)

        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            if args["verbose"]:
                print("Intersection:", accepted_features)
            self.coefficients = np.zeros(dim)

            if len(accepted_features):
                x_s = x_all[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y_all)
                self.coefficients[list(accepted_features)] = reg.coef_

            self.coefficients = torch.Tensor(self.coefficients)
        else:
            self.coefficients = torch.zeros(dim)

    def mean_var_test(self, x, y):
        pvalue_mean = ttest_ind(x, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(x, ddof=1) / np.var(y, ddof=1),
                                    x.shape[0] - 1,
                                    y.shape[0] - 1)

        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s):
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def solution(self):
        return self.coefficients


class EmpiricalRiskMinimizer(object):
    def __init__(self, environments, args):
        x_all = torch.cat([x for (x, y) in environments]).numpy()
        y_all = torch.cat([y for (x, y) in environments]).numpy()

        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w)

    def solution(self):
        return self.w
