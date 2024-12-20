import torch
import torch.nn as nn


class VAELossorg(nn.Module):
    def __init__(self):
        super(VAELossorg, self).__init__()

    def forward(self, x, recon_x, mu, logvar):

        recon_loss =  torch.mean(0.5*(x - recon_x).pow(2))

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
        
        # Total loss
        total_loss = recon_loss + kl_loss * 1e-7
        log = {
            "total_loss": total_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "rec_loss": recon_loss.detach()
        }
        return total_loss, log    
    

#################
##NOT USED#######
#################








class LPIPSLossorg(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", deterministic=False):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.deterministic = deterministic
        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm
        #                                          ).apply(weights_init)
        # self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional

    # def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
    #     if last_layer is not None:
    #         nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    #     else:
    #         nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
    #         g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

    #     d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    #     d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    #     d_weight = d_weight * self.discriminator_weight
    #     return d_weight

    def kl(self, mean, logstd, mean2=None, logstd2=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if mean2 is None:
                return 0.5 * torch.sum(torch.pow(mean, 2)
                                       + torch.exp(logstd) - 1.0 - logstd,
                                       dim=-1)
            else:
                return 0.5 * torch.sum(
                    torch.pow(mean - mean2, 2) / torch.exp(logstd)
                    + torch.exp(logstd) / torch.exp(logstd2)  - 1.0 - logstd + logstd2,
                    dim=-1)


    def forward(self, inputs, reconstructions, mean, logstd, optimizer_idx=None,
                global_step=None, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # if self.perceptual_weight > 0:
        #     p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        #     rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        
        
        
        kl_loss = self.kl(mean, logstd)
        
        
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = weighted_nll_loss + self.kl_weight * kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
               "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), 
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
            #    "{}/d_weight".format(split): d_weight.detach(),
            #    "{}/disc_factor".format(split): torch.tensor(disc_factor),
            #    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
        return loss, log

class VAELoss(nn.Module):
    def __init__(self, rec_logvar_init=0.0, kl_weight=0.000001, pixelloss_weight=1.0, deterministic=False):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * rec_logvar_init)
        self.deterministic = deterministic

    def kl(self, mean, logvar, mean2=None, logvar2=None):
        if self.deterministic:
            return torch.tensor(0., device=mean.device)
        else:
            if mean2 is None:
                return 0.5 * torch.sum(
                    mean.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=-1
                )
            else:
                return 0.5 * torch.sum(
                    (logvar2 - logvar) +
                    (torch.exp(logvar) + (mean - mean2).pow(2)) / torch.exp(logvar2) - 1.0,
                    dim=-1
                )

    def forward(self, inputs, reconstructions, mean, logvar, weights=None):
        # Reconstruction loss (MSE)
        rec_loss = self.pixel_weight * (inputs - reconstructions).pow(2)
        # Negative Log-Likelihood Loss
        nll_loss = 0.5 * (rec_loss / torch.exp(self.logvar) + self.logvar)
        if weights is not None:
            nll_loss = weights * nll_loss
        # Compute mean loss over batch and elements
        nll_loss = torch.mean(nll_loss)
        # KL Divergence Loss
        kl_loss = self.kl(mean, logvar)
        kl_loss = torch.mean(kl_loss)
        # Total Loss
        loss = nll_loss + self.kl_weight * kl_loss
        # Logging
        log = {
            "total_loss": loss.detach(),
            "logvar": self.logvar.detach(),
            "kl_loss": kl_loss.detach(),
            "nll_loss": nll_loss.detach(),
            "rec_loss": torch.mean(rec_loss).detach(),
        }
        return loss, log
    

    

class LPIPSLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 perceptual_weight=1.0, deterministic=False):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.deterministic = deterministic

        # Initialize LPIPS perceptual loss model
        # self.perceptual_loss = LPIPS().eval()

        # Output log variance (learnable parameter)
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def kl(self, mean, logstd, mean2=None, logstd2=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if mean2 is None:
                return 0.5 * torch.sum(
                    torch.pow(mean, 2) + torch.exp(2*logstd) - 1.0 - 2*logstd, dim=-1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(mean - mean2, 2) / torch.exp(logstd2)
                    + torch.exp(logstd) / torch.exp(logstd2)
                    - 1.0
                    - logstd
                    + logstd2,
                    dim=-1,
                )

    def forward(self, inputs, reconstructions, mean, logstd, weights=None):
        # Ensure inputs and reconstructions are contiguous and on the same device
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # Compute pixel-wise reconstruction loss (Mean Squared Error)
        rec_loss = self.pixel_weight * (inputs - reconstructions).pow(2)

        # Compute perceptual loss using LPIPS
        # if self.perceptual_weight > 0:
        #     # print("input", inputs.shape)
        #     p_loss = self.perceptual_loss(inputs, reconstructions)
        #     # Ensure perceptual loss has the same dimensions as rec_loss
        #     # print("rec_loss",p_loss,p_loss.shape)
        #     # p_loss = p_loss.view_as(rec_loss)
        #     rec_loss += self.perceptual_weight * p_loss

        # Negative Log-Likelihood Loss (assuming Gaussian likelihood)
        nll_loss = 0.5 * (rec_loss / torch.exp(self.logvar) + self.logvar)

        # Apply weights if provided
        if weights is not None:
            nll_loss *= weights

        # Compute mean over all elements
        nll_loss = torch.mean(nll_loss)

        # Compute KL Divergence Loss
        kl_loss = torch.mean(self.kl(mean, logstd))

        # Total Loss
        loss = nll_loss + self.kl_weight * kl_loss

        # Logging (detach to prevent gradients)
        log = {
            "total_loss": loss.detach(),
            # "logvar": self.logvar.detach(),
            "kl_loss": kl_loss.detach(),
            # "nll_loss": nll_loss.detach(),
            "rec_loss": rec_loss.mean().detach(),
        }

        return loss, log