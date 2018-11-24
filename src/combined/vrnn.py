"""Multimodal Variational Recurrent Neural Network, adapted from
https://github.com/emited/VariationalRecurrentNeuralNetwork

Original VRNN described in https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

class VRNN(nn.Module):
    def __init__(self, audio_dim=990, text_dim=300, visual_dim=4096,
                 h_dim=128, z_dim=128, n_layers=1, bias=False,
                 lambda_audio=1.0, lambda_text=1.0, lambda_visual=1.0,
                 use_cuda=False):
        super(VRNN, self).__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        # Loss multipliers
        self.lambda_audio = lambda_audio
        self.lambda_text = lambda_text
        self.lambda_visual = lambda_visual

        # Feature-extracting transformations
        self.phi_audio = nn.Sequential(
            nn.Linear(audio_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_text = nn.Sequential(
            nn.Linear(text_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_visual = nn.Sequential(
            nn.Linear(visual_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # Encoder p(z|x) = N(mu(x,h), sigma(x,h))
        self.enc = nn.Sequential(
            nn.Linear(3*h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # Prior p(z) = N(mu(h), sigma(h))
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # Decoders p(xi|z) = N(mu(z,h), sigma(z,h))
        self.dec_audio = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_audio_std = nn.Sequential(
            nn.Linear(h_dim, audio_dim),
            nn.Softplus())
        self.dec_audio_mean = nn.Linear(h_dim, audio_dim)

        self.dec_text = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_text_std = nn.Sequential(
            nn.Linear(h_dim, text_dim),
            nn.Softplus())
        self.dec_text_mean = nn.Linear(h_dim, text_dim)

        self.dec_visual = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_visual_std = nn.Sequential(
            nn.Linear(h_dim, visual_dim),
            nn.Softplus())
        self.dec_visual_mean = nn.Sequential(
            nn.Linear(h_dim, visual_dim),
            nn.Sigmoid())

        # Recurrence h_next = f(x,z,h)
        self.rnn = nn.GRU(3*h_dim + h_dim, h_dim, n_layers, bias)

        if use_cuda:
            self.cuda()
        
    def forward(self, audio, text, visual):
        seq_length = audio.size(0)
        batch_size = audio.size(1)
        
        z_mean, z_std = [], []
        audio_mean, audio_std = [], []
        text_mean, text_std = [], []
        visual_mean, visual_std = [], []

        kld_loss = 0
        nll_loss = 0

        # Initialize hidden state
        h = torch.zeros(self.n_layers, batch_size, self.h_dim)
        if self.use_cuda:
            h = h.cuda()
            
        for t in range(seq_length):
            # Extract features from inputs
            phi_audio_t = self.phi_audio(audio[t])
            phi_text_t = self.phi_text(text[t])
            phi_visual_t = self.phi_visual(visual[t])
            phi_all_t = torch.cat([phi_audio_t, phi_text_t, phi_visual_t], 1)
            
            # Encode features to latent code z
            enc_t = self.enc(torch.cat([phi_all_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # Compute prior for z
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sample and reparameterize
            z_t = self._sample_gauss(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_z_t, h[-1]], 1)

            dec_audio_t = self.dec_audio(dec_in)
            dec_audio_mean_t = self.dec_audio_mean(dec_t)
            dec_audio_std_t = self.dec_audio_std(dec_t)

            dec_text_t = self.dec_text(dec_in)
            dec_text_mean_t = self.dec_text_mean(dec_t)
            dec_text_std_t = self.dec_text_std(dec_t)

            dec_visual_t = self.dec_visual(dec_in)
            dec_visual_mean_t = self.dec_visual_mean(dec_t)
            dec_visual_std_t = self.dec_visual_std(dec_t)

            # Recurrence
            _, h = self.rnn(torch.cat([phi_all_t, phi_z_t], 1).unsqueeze(0), h)

            # Compute losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t,
                                        prior_mean_t, prior_std_t)
            nll_loss += self.lambda_audio *\
                self._nll_gauss(dec_audio_mean_t, dec_audio_std_t, audio[t])
            nll_loss += self.lambda_text *\
                self._nll_gauss(dec_text_mean_t, dec_text_std_t, text[t])
            nll_loss += self.lambda_visual *\
                self._nll_gauss(dec_visual_mean_t, dec_visual_std_t, visual[t])

            # Save inferred latent z and reconstructed outputs
            z_mean.append(enc_mean_t)
            z_std.append(enc_std_t)
            audio_mean.append(dec_audio_mean_t)
            audio_std.append(dec_audio_std_t)
            text_mean.append(dec_text_mean_t)
            text_std.append(dec_text_std_t)
            visual_mean.append(dec_visual_mean_t)
            visual_std.append(dec_visual_std_t)

        z_infer = (z_mean, z_std)
        recon = (audio_mean, audio_std,
                 text_mean, text_std,
                 visual_mean, visual_std) 
        return kld_loss, nll_loss, z_infer, recon            


    def sample(self, seq_length):
        audio = torch.zeros(seq_length, self.audio_dim)
        text = torch.zeros(seq_length, self.text_dim)
        visual = torch.zeros(seq_length, self.visual_dim)
        h = torch.zeros(self.n_layers, 1, self.h_dim)

        if self.use_cuda:
            audio = audio.cuda()
            text = text.cuda()
            visual = visual.cuda()
            h = h.cuda()
        for t in range(seq_length):
            # Compute prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sample from prior
            z_t = self._sample_gauss(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
            
            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_z_t, h[-1]], 1)

            dec_audio_t = self.dec_audio(dec_in)
            dec_audio_mean_t = self.dec_audio_mean(dec_t)
            dec_audio_std_t = self.dec_audio_std(dec_t)

            dec_text_t = self.dec_text(dec_in)
            dec_text_mean_t = self.dec_text_mean(dec_t)
            dec_text_std_t = self.dec_text_std(dec_t)

            dec_visual_t = self.dec_visual(dec_in)
            dec_visual_mean_t = self.dec_visual_mean(dec_t)
            dec_visual_std_t = self.dec_visual_std(dec_t)

            # Extract features from inputs
            phi_audio_t = self.phi_audio(dec_audio_mean_t)
            phi_text_t = self.phi_text(dec_text_mean_t)
            phi_visual_t = self.phi_visual(dec_visual_mean_t)
            phi_all_t = torch.cat([phi_audio_t, phi_text_t, phi_visual_t], 1)
            
            # Recurrence
            _, h = self.rnn(torch.cat([phi_all_t, phi_z_t], 1).unsqueeze(0), h)

            audio[t] = dec_audio_mean_t
            text[t] = dec_text_mean_t
            visual[t] = dec_visual_mean_t
    
        return audio, text, visual

    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Use std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return  0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _nll_gauss(self, mean, std, x):
        return ( ((x-mean).pow(2)) / (2 * std.pow(2)) + std.log() +
                 math.log(math.sqrt(2 * math.pi)) )
