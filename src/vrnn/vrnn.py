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

class VRNN(nn.Module):
    def __init__(self, audio_dim=990, text_dim=300, visual_dim=4096,
                 h_dim=128, z_dim=128, n_layers=1, bias=False, use_cuda=False):
        super(VRNN, self).__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.use_cuda = use_cuda

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
        self.dec_visual_mean = nn.Linear(h_dim, visual_dim)

        # Decoder to predict valence
        self.dec_val = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_val_std = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Softplus())
        self.dec_val_mean = nn.Linear(h_dim, 1)
        
        # Recurrence h_next = f(x,z,h)
        self.rnn = nn.GRU(3*h_dim + h_dim, h_dim, n_layers, bias)

        if use_cuda:
            self.cuda()

            
    def forward(self, audio, text, visual):
        seq_length = audio.size(0)
        batch_size = audio.size(1)

        # Initialize list accumulators
        prior_mean, prior_std = [], []
        infer_mean, infer_std = [], []

        audio_mean, audio_std = [], []
        text_mean, text_std = [], []
        visual_mean, visual_std = [], []

        val_mean, val_std = [], []
        
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
            infer_t = self.enc(torch.cat([phi_all_t, h[-1]], 1))
            infer_mean.append(self.enc_mean(infer_t))
            infer_std.append(self.enc_std(infer_t))

            # Compute prior for z
            prior_t = self.prior(h[-1])
            prior_mean.append(self.prior_mean(prior_t))
            prior_std.append(self.prior_std(prior_t))

            # Sample and reparameterize
            z_t = self._sample_gauss(infer_mean[t], infer_std[t])
            phi_z_t = self.phi_z(z_t)

            # Decode sampled z to reconstruct inputs
            dec_in_t = torch.cat([phi_z_t, h[-1]], 1)

            audio_t = self.dec_audio(dec_in_t)
            audio_mean.append(self.dec_audio_mean(audio_t))
            audio_std.append(self.dec_audio_std(audio_t))

            text_t = self.dec_text(dec_in_t)
            text_mean.append(self.dec_text_mean(text_t))
            text_std.append(self.dec_text_std(text_t))

            visual_t = self.dec_visual(dec_in_t)
            visual_mean.append(self.dec_visual_mean(visual_t))
            visual_std.append(self.dec_visual_std(visual_t))

            # Decode z to predict valence
            val_t = self.dec_val(dec_in_t)
            val_mean.append(self.dec_val_mean(val_t))
            val_std.append(self.dec_val_std(val_t))
            
            # Recurrence
            _, h = self.rnn(torch.cat([phi_all_t, phi_z_t], 1).unsqueeze(0), h)

        infer = (torch.stack(infer_mean), torch.stack(infer_std))
        prior = (torch.stack(prior_mean), torch.stack(prior_std))
        recon = (torch.stack(audio_mean), torch.stack(audio_std),
                 torch.stack(text_mean), torch.stack(text_std),
                 torch.stack(visual_mean), torch.stack(visual_std))
        val = (torch.stack(val_mean), torch.stack(val_std))
        return infer, prior, recon, val

    
    def sample(self, seq_length):
        audio_mean = torch.zeros(seq_length, self.audio_dim)
        text_mean = torch.zeros(seq_length, self.text_dim)
        visual_mean = torch.zeros(seq_length, self.visual_dim)
        val_mean = torch.zeros(seq_length, 1)
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

            audio_t = self.dec_audio(dec_in_t)
            audio_mean[t] = self.dec_audio_mean(audio_t)
            audio_std_t = self.dec_audio_std(audio_t)

            text_t = self.dec_text(dec_in_t)
            text_mean[t] = self.dec_text_mean(text_t)
            text_std_t = self.dec_text_std(text_t)

            visual_t = self.dec_visual(dec_in_t)
            visual_mean[t] = self.dec_visual_mean(visual_t)
            visual_std_t = self.dec_visual_std(visual_t)

            # Decode z to predict valence
            val_t = self.dec_val(dec_in_t)
            val_mean[t] = self.dec_val_mean(val_t)
            val_std_t = self.dec_val_std(val_t)
            
            # Extract features from inputs
            phi_audio_t = self.phi_audio(audio_mean[t])
            phi_text_t = self.phi_text(text_mean[t])
            phi_visual_t = self.phi_visual(visual_mean[t])
            phi_all_t = torch.cat([phi_audio_t, phi_text_t, phi_visual_t], 1)
            
            # Recurrence
            _, h = self.rnn(torch.cat([phi_all_t, phi_z_t], 1).unsqueeze(0), h)
    
        return audio_mean, text_mean, visual_mean, val_mean


    def loss(self, inputs, val_obs, infer, prior, recon, val, mask=1,
             kld_mult=1.0, rec_mults=(1.0, 1.0, 1.0), sup_mult=1.0, avg=False):
        loss = 0.0
        loss += kld_mult * self.kld_loss(infer, prior, mask)
        loss += self.rec_loss(inputs, recon, mask, rec_mults)
        loss += sup_mult * self.sup_loss(val, val_obs, mask)
        if avg:
            if type(mask) is torch.Tensor:
                n_data = torch.sum(mask)
            else:
                n_data = val_obs.numel()
            loss /= n_data
        return loss

    
    def kld_loss(self, infer, prior, mask=None):
        """KLD loss between inferred and prior z."""
        infer_mean, infer_std = infer
        prior_mean, prior_std = prior
        return self._kld_gauss(infer_mean, infer_std,
                               prior_mean, prior_std, mask)

    
    def rec_loss(self, inputs, recon, mask=None, lambdas=(1.0, 1.0, 1.0)):
        """Input reconstruction loss."""
        audio, text, visual = inputs
        audio_mean, audio_std = recon[:2]
        text_mean, text_std = recon[2:4]
        visual_mean, visual_std = recon[4:]
        l_audio, l_text, l_visual = lambdas

        loss = 0.0
        loss += l_audio*self._nll_gauss(audio_mean, audio_std, audio, mask)
        loss += l_text*self._nll_gauss(text_mean, text_std, text, mask)
        loss += l_visual*self._nll_gauss(visual_mean, visual_std, visual, mask)
        return loss

    
    def sup_loss(self, val_params, val_obs, mask=None):
        """Supervised prediction loss."""
        val_mean, val_std = val_params
        return self._nll_gauss(val_mean, val_std, val_obs, mask)
    
    
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _sample_gauss(self, mean, std):
        """Use std to sample."""
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = eps.cuda()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, mask=None):
        """Use std to compute KLD"""
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        if mask is not None:
            kld_element = kld_element.masked_select(mask)
        return  0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x, mask=None):
        nll_element = x*torch.log(theta) + (1-x)*torch.log(1-theta)
        if mask is not None:
            nll_element = nll_element.masked_select(mask)
        return torch.sum(nll_element)


    def _nll_gauss(self, mean, std, x, mask=None):
        nll_element = ( ((x-mean).pow(2)) / (2 * std.pow(2)) + std.log() +
                        math.log(math.sqrt(2 * math.pi)) )
        if mask is not None:
            nll_element = nll_element.masked_select(mask)
        return torch.sum(nll_element)

    
if __name__ == "__main__":
    # Test code by loading dataset and running through model
    import os
    from datasets import OMGcombined
    
    base_folder = "./data/Training"
    audio_path = os.path.join(base_folder, "CombinedAudio")
    text_path = os.path.join(base_folder, "CombinedText")
    visual_path = os.path.join(base_folder, "CombinedVisual")
    valence_path = os.path.join(base_folder, "Annotations")

    print("Loading data...")
    dataset = OMGcombined(audio_path, text_path, visual_path, valence_path)

    print("Building model...")
    model = VRNN()
    model.eval()

    print("Passing a sample through the model...")
    audio, text, visual, valence = dataset[0]
    audio = torch.tensor(audio).unsqueeze(1).float()
    text = torch.tensor(text).unsqueeze(1).float()
    visual = torch.tensor(visual).unsqueeze(1).float()
    val_obs = torch.tensor(valence).unsqueeze(1).float()

    infer, prior, recon, val = model(audio, text, visual)
    val_mean, val_std = val
    loss = model.loss((audio, text, visual), val_obs,
                      infer, prior, recon, val)
    print("Average loss: {:0.3f}".format(loss))
    print("Predicted valences:")
    for v in val_mean:
        print("{:+0.3f}".format(v.item()))
