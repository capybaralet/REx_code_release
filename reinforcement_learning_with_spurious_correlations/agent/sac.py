import numpy as np
import torch
import torch.nn as nn
from torch import optim, autograd
import torch.nn.functional as F
import math


from agent import Agent
import utils
from encoder import make_encoder
from decoder import make_decoder

import hydra


OUT_DIM = {2: 39, 4: 35, 6: 31}


def make_dynamics_model(feature_dim, hidden_dim, action_shape):
    model = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim))
    return model


def irm_penalty(logits, labels):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = F.mse_loss(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


class RewardModel(nn.Module):
    def __init__(self, representation_size, action_shape):
        super().__init__()

        self.action_linear = nn.Linear(action_shape, representation_size)
        self.trunk = nn.Sequential(
            nn.Linear(representation_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, state, action):
        action_emb = self.action_linear(action)
        return self.trunk(torch.cat([state, action_emb], dim=-1))


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


class RrexAgent(Agent):
    """Rrex algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, encoder_type,
                 critic_cfg, actor_cfg, discount, init_temperature, alpha_lr, l2_regularizer_weight,
                 alpha_betas, actor_lr, actor_betas, actor_update_frequency,
                 critic_lr, critic_betas, critic_tau, num_envs, encoder_feature_dim, penalty_type,
                 critic_target_update_frequency, encoder_batch_size, sac_batch_size, penalty_anneal_iters, 
                 penalty_weight):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.encoder_batch_size = encoder_batch_size
        self.sac_batch_size = sac_batch_size
        self.num_envs = num_envs
        self.l2_regularizer_weight = l2_regularizer_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.penalty_weight = penalty_weight
        self.penalty_type = penalty_type

        self.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2,
            32
        ).to(self.device)
        self.reward_model = nn.Sequential(
            nn.Linear(encoder_feature_dim, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 1)
        ).to(device)
        self.model = make_dynamics_model(encoder_feature_dim, 200, action_dim).to(device)
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic.encoder = self.encoder

        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, 2,
            32
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor.encoder = self.encoder

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(
                self.reward_model.parameters()) + list(self.encoder.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger, step):
        dist = self.actor(next_obs, detach=False)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action, detach=True)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2, h = self.critic(obs, action, return_latent=True, detach=False)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)
        
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # add L1 penalty
        L1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for name, param in self.critic.encoder.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        return critic_loss + self.l2_regularizer_weight * L1_reg, target_V.mean()

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs, detach=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        return actor_loss, alpha_loss

    def update_decoder(self, obs, action, reward, target_obs, logger, step, i):
        h = self.critic.encoder(obs[0])
        
        r_hats = []
        for i  in range(self.seq_len):
            h =  self.model(torch.cat([h, action[i]], dim=-1))
            r_hats.append(self.reward_model(h))

        r_hats = torch.stack(r_hats)
        rew_loss = F.mse_loss(r_hats, reward)
        if self.penalty_type == 'irm':
            self.irm_penalty =  irm_penalty(r_hats, reward)

        # autoencoder loss
        # rec_obs = self.decoder(torch.cat([h, task_specific_h], dim=-1))
        # rec_loss = F.mse_loss(obs, rec_obs)


        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        # add L1 penalty
        L1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for name, param in self.critic.encoder.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        logger.log('train/reward_loss', rew_loss, step)
        return rew_loss

    def update(self, replay_buffer, logger, step, train_sac=True, train_encoder=True):
        total_actor_loss, total_alpha_loss, total_critic_loss = [], [], []
        target_vs = []
        decoder_loss = []
        irm_penalties = []
        if train_encoder:
            batch_size = self.encoder_batch_size
        else:
            batch_size = self.sac_batch_size
        for env_id in range(self.num_envs):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                batch_size, env_id)

            logger.log('train/batch_reward', reward.mean(), step)
            if train_sac:
                critic_loss, target_v = self.update_critic(
                    obs.view(-1, *(obs.shape[2:])), 
                    action.view(-1, *(action.shape[2:])), 
                    reward.view(-1, *(reward.shape[2:])), 
                    next_obs.view(-1, *(next_obs.shape[2:])), 
                    not_done_no_max.view(-1, *(not_done_no_max.shape[2:])),
                                logger, step)
                total_critic_loss.append(critic_loss)
                target_vs.append(target_v)

                if step % self.actor_update_frequency == 0:
                    actor_loss, alpha_loss = self.update_actor_and_alpha(obs.view(-1, *(obs.shape[2:])), logger, step)
                    total_actor_loss.append(actor_loss)
                    total_alpha_loss.append(alpha_loss)

            if train_encoder:
                decoder_loss.append(self.update_decoder(obs, action, reward, next_obs, logger, step, env_id))
            if self.penalty_type == 'irm':
                irm_penalties.append(self.irm_penalty)

        if train_encoder:
            self.decoder_optimizer.zero_grad()
            if self.penalty_type == 'rex':
                train_penalty = torch.abs(decoder_loss[0] - decoder_loss[1])  # hardcoded for 2 envs
            elif self.penalty_type == 'irm':
                train_penalty = torch.stack(irm_penalties).mean()
            else:  # assume penalty_type is rem
                train_penalty = 0
            decoder_loss = torch.stack(decoder_loss).mean()
            penalty_weight = (self.penalty_weight 
                if step >= self.penalty_anneal_iters else 1.0)
            logger.log('train_encoder/penalty', train_penalty, step)
            decoder_loss += penalty_weight * train_penalty
            if penalty_weight > 1.0 and self.penalty_type != 'erm':
                # Rescale the entire loss to keep gradients in a reasonable range
                decoder_loss /= penalty_weight

            decoder_loss.backward()

            self.decoder_optimizer.step()
            
        # Optimize the critic
        if train_sac:
            total_critic_loss = torch.stack(total_critic_loss).mean()
            
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer.step()
            self.critic.log(logger, step)

            if step % self.actor_update_frequency == 0:
                # optimize the actor
                self.actor_optimizer.zero_grad()
                torch.stack(total_actor_loss).mean().backward()
                self.actor_optimizer.step()

                self.actor.log(logger, step)

                self.log_alpha_optimizer.zero_grad()
                torch.stack(total_alpha_loss).mean().backward()
                self.log_alpha_optimizer.step()

            if step % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                        self.critic_tau)