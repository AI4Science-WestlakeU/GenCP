# Implementation of the filters is borrowed from paper "Alias-Free Generative Adversarial Networks (StyleGAN3)" https://nvlabs.github.io/stylegan3/
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#-------------------------------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from model.CNO_libs.training.filtered_networks import LReLu #Either "filtered LReLU" or regular LReLu
import pdb
#------------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Encode timestep t as sinusoidal positional encoding
        Args:
            t: [batch_size] timestep
        Returns:
            time_emb: [batch_size, dim] time embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class FiLM(nn.Module):
    def __init__(self, time_dim, channels):
        super().__init__()
        self.scale_shift = nn.Linear(time_dim, 2 * channels)
        
    def forward(self, x, time_emb):
        """
        Modulate features using time embedding
        Args:
            x: [B, C, T, H, W] feature map
            time_emb: [B, time_dim] time embedding
        Returns:
            Modulated feature map
        """
        scale_shift = self.scale_shift(time_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
        shift = shift.view(shift.shape[0], shift.shape[1], 1, 1, 1)
        return x * (1 + scale) + shift

#------------------------------------------------------------------------------

#Depending on in_size, out_size, the CNOBlock can be:
#   -- (D) Block
#   -- (U) Block
#   -- (I) Block

class CNOBlock3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',
                 time_dim = None
                 ):
        super(CNOBlock3d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        
        self.in_halfwidth =  half_width_mult*self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den
        
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv3d(in_channels = self.in_channels, out_channels=self.out_channels, 
                                           kernel_size=self.conv_kernel, 
                                           padding = pad)
    
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm3d(self.out_channels)

        self.time_dim = time_dim
        if time_dim is not None:
            self.time_film = FiLM(time_dim, self.out_channels)

        if activation == 'lrelu':
            self.activation  = LReLu(in_channels           = self.in_channels, #In _channels is not used in these settings
                                        out_channels          = self.out_channels,                   
                                        in_size               = self.in_size,                       
                                        out_size              = self.out_size,                       
                                        in_sampling_rate      = self.in_size,               
                                        out_sampling_rate     = self.out_size,             
                                        in_cutoff             = self.in_cutoff,                     
                                        out_cutoff            = self.out_cutoff,                  
                                        in_half_width         = self.in_halfwidth,                
                                        out_half_width        = self.out_halfwidth,              
                                        filter_size           = filter_size,       
                                        lrelu_upsampling      = lrelu_upsampling,
                                        is_critically_sampled = self.citically_sampled,
                                        use_radial_filters    = False)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
    def forward(self, x, time_emb=None):
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        
        if self.time_dim is not None and time_emb is not None:
            x = self.time_film(x, time_emb)
            
        return self.activation(x)

#------------------------------------------------------------------------------

# Contains CNOBlock -> Convolution -> BN

class LiftProjectBlock3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',
                 time_dim = None
                 ):
        super(LiftProjectBlock3d, self).__init__()
    
        self.inter_CNOBlock = CNOBlock3d(in_channels = in_channels,
                                    out_channels = latent_dim,
                                    in_size = in_size,
                                    out_size = out_size,
                                    cutoff_den = cutoff_den,
                                    conv_kernel = conv_kernel,
                                    filter_size = filter_size,
                                    lrelu_upsampling = lrelu_upsampling,
                                    half_width_mult  = half_width_mult,
                                    radial = radial,
                                    batch_norm = batch_norm,
                                    activation = activation,
                                    time_dim = time_dim)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv3d(in_channels = latent_dim, out_channels=out_channels, 
                                           kernel_size=conv_kernel, stride = 1, 
                                           padding = pad)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm3d(out_channels)
        
        self.time_dim = time_dim
        if time_dim is not None:
            self.time_film = FiLM(time_dim, out_channels)
        
    def forward(self, x, time_emb=None):
        x = self.inter_CNOBlock(x, time_emb)
        
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
            
        if self.time_dim is not None and time_emb is not None:
            x = self.time_film(x, time_emb)
            
        return x
        
#------------------------------------------------------------------------------

# Residual Block containts:
    # Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock3d(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',
                 time_dim = None
                 ):
        super(ResidualBlock3d, self).__init__()

        self.channels = channels
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv3d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        self.convolution2 = torch.nn.Conv3d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm3d(self.channels)
            self.batch_norm2  = nn.BatchNorm3d(self.channels)
        
        self.time_dim = time_dim
        if time_dim is not None:
            self.time_film1 = FiLM(time_dim, self.channels)
            self.time_film2 = FiLM(time_dim, self.channels)
        
        if activation == 'lrelu':
            self.activation  = LReLu(in_channels           = self.channels, #In _channels is not used in these settings
                                     out_channels          = self.channels,                   
                                     in_size               = self.size,                       
                                     out_size              = self.size,                       
                                     in_sampling_rate      = self.size,               
                                     out_sampling_rate     = self.size,             
                                     in_cutoff             = self.cutoff,                     
                                     out_cutoff            = self.cutoff,                  
                                     in_half_width         = self.halfwidth,                
                                     out_half_width        = self.halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Activation function {activation} not supported")
            

    def forward(self, x, time_emb=None):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        
        if self.time_dim is not None and time_emb is not None:
            out = self.time_film1(out, time_emb)
            
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        
        if self.time_dim is not None and time_emb is not None:
            out = self.time_film2(out, time_emb)
        
        return x + out
#------------------------------------------------------------------------------

#CNO NETWORK (Without x0):
class CNO3d(nn.Module):
    def __init__(self,  
                 in_dim,                    # Number of input channels.
                 in_size,                   # Input spatial size
                 N_layers,                  # Number of (D) or (U) blocks in the network
                 N_res = 1,                 # Number of (R) blocks per level (except the neck)
                 N_res_neck = 6,            # Number of (R) blocks in the neck
                 channel_multiplier = 32,   # How the number of channels evolve?
                 conv_kernel=3,             # Size of all the kernels
                 cutoff_den = 2.0001,       # Filter property 1.
                 filter_size=6,             # Filter property 2.
                 lrelu_upsampling = 2,      # Filter property 3.
                 half_width_mult  = 0.8,    # Filter property 4.
                 radial = False,            # Filter property 5. Is filter radial?
                 batch_norm = True,         # Add BN? We do not add BN in lifting/projection layer
                 out_dim = 1,               # Target dimension
                 out_dim_mult = 1,          # If out_dim_mult is 1, Then out_dim = out_dim. Else must be int
                 out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                 expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                 latent_lift_proj_dim = 64, # Intermediate latent dimension in the lifting/projection layer
                 add_inv = True,            # Add invariant block (I) after the intermediate connections?
                 activation = 'LeakyReLU',   # Activation function can be 'LeakyReLU' or 'lrelu'
                 dataset_name = "turek_hron_data",
                 x0_is_use_noise = True,
                 stage = "fuel",
                 time_dim = 128
               ):
        
        super(CNO3d, self).__init__()

        ###################### Define the parameters & specifications #################################################

        
        # Number od (D) & (U) Blocks
        self.N_layers = int(N_layers)
        
        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2
        self.out_dim_mult = out_dim_mult
        self.out_dim  = out_dim * out_dim_mult
        
        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        
        
        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True
        
        self.dataset_name = dataset_name
        self.x0_is_use_noise = x0_is_use_noise
        self.stage = stage
        self.time_dim = time_dim
        
        if time_dim is not None:
            self.time_embedding = TimeEmbedding(time_dim)
        
        ###################### Define evolution of the number features ################################################

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i * self.channel_multiplier)
        
        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])
        
        ###################### Define evolution of sampling rates #####################################################
        
        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
        
        
        
        ###################### Define Projection & Lift ##############################################################
    
        self.lift = LiftProjectBlock3d(in_channels  = in_dim,
                                     out_channels = self.encoder_features[0],
                                     in_size      = in_size,
                                     out_size     = self.encoder_sizes[0],
                                     latent_dim   = latent_lift_proj_dim,
                                     cutoff_den   = cutoff_den,
                                     conv_kernel  = conv_kernel,
                                     filter_size  = filter_size,
                                     lrelu_upsampling  = lrelu_upsampling,
                                     half_width_mult   = half_width_mult,
                                     radial            = radial,
                                     batch_norm        = False,
                                     activation = activation,
                                     time_dim = time_dim)
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
            
        self.project = LiftProjectBlock3d(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = self.out_dim,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        radial            = radial,
                                        batch_norm        = False,
                                        activation = activation,
                                        time_dim = time_dim) 

        ###################### Define U & D blocks ###################################################################

        self.encoder         = nn.ModuleList([(CNOBlock3d(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        time_dim = time_dim))                                  
                                               for i in range(self.N_layers)])
        
        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock3d(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        time_dim = time_dim))                                  
                                               for i in range(self.N_layers + 1)])
        
        self.decoder         = nn.ModuleList([(CNOBlock3d(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        time_dim = time_dim))                                  
                                               for i in range(self.N_layers)])
        
        
        self.decoder_inv    = nn.ModuleList([(CNOBlock3d(in_channels  =  self.inv_features[i],
                                                        out_channels = self.inv_features[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,
                                                        time_dim = time_dim))                                  
                                               for i in range(self.N_layers + 1)])
        
        
        ####################### Define ResNets Blocks ################################################################

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        res_nets_list = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                res_nets_list.append(ResidualBlock3d(channels = self.encoder_features[l],
                                                   size     = self.encoder_sizes[l],
                                                   cutoff_den = cutoff_den,
                                                   conv_kernel = conv_kernel,
                                                   filter_size = filter_size,
                                                   lrelu_upsampling = lrelu_upsampling,
                                                   half_width_mult  = half_width_mult,
                                                   radial = radial,
                                                   batch_norm = batch_norm,
                                                   activation = activation,
                                                   time_dim = time_dim))
        for i in range(self.N_res_neck):
            res_nets_list.append(ResidualBlock3d(channels = self.encoder_features[self.N_layers],
                                               size     = self.encoder_sizes[self.N_layers],
                                               cutoff_den = cutoff_den,
                                               conv_kernel = conv_kernel,
                                               filter_size = filter_size,
                                               lrelu_upsampling = lrelu_upsampling,
                                               half_width_mult  = half_width_mult,
                                               radial = radial,
                                               batch_norm = batch_norm,
                                               activation = activation,
                                               time_dim = time_dim))
        
        self.res_nets = nn.ModuleList(res_nets_list)    

    def _forward_impl(self, x, t):
        """
        Core forward implementation that maps (B, T, H, W, C) -> (B, T, H, W, C_out).
        This matches the original NTcouple forward behavior in this repository.
        """
        switch = False
        if x.dim() == 5 and x.shape[-1] < x.shape[1]:
            switch = True
            x = x.permute(0, 4, 1, 2, 3)
        
        time_emb = None
        if self.time_dim is not None and t is not None:
            if t.dim() > 1:
                t = t.squeeze()
            time_emb = self.time_embedding(t)
                 
        x = self.lift(x, time_emb)
        skip = []
        
        for i in range(self.N_layers):
            y = x
            for j in range(self.N_res):
                y = self.res_nets[i*self.N_res + j](y, time_emb)
            skip.append(y)
            x = self.encoder[i](x, time_emb)
        
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x, time_emb)

        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x, time_emb)
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i], time_emb)), 1)
            
            if self.add_inv:
                x = self.decoder_inv[i](x, time_emb)
            x = self.decoder[i](x, time_emb)
        
        x = torch.cat((x, self.ED_expansion[0](skip[0], time_emb)), 1)
        x = self.project(x, time_emb)
        
        del skip
        del y

        if switch:
            x = x.permute(0, 2, 3, 4, 1)

        if self.out_dim_mult > 1:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.out_dim // self.out_dim_mult)
        
        return x

    def forward(self, x, t, x0_or_cond=None, cond=None, **kwargs):
        """
        Compatible forward for both FSI and NTcouple.
        
        - FSI:        forward(xt, t, x0, cond)   (or keyword x0=..., cond=...)
        - NTcouple:   forward(x_joint, t, cond)  (cond can be passed as 3rd positional or as cond=...)
        """
        # Backward compatibility for callers using keyword arguments:
        if x0_or_cond is None and "x0" in kwargs:
            x0_or_cond = kwargs.pop("x0")
        if cond is None and "cond" in kwargs:
            cond = kwargs.pop("cond")

        if self.dataset_name == "ntcouple":
            # NTcouple: keep original behavior; allow passing attrs either as 3rd positional or cond=...
            if x0_or_cond is not None:
                cond = x0_or_cond
            return self._forward_impl(x, t)

        # FSI: prepend x0 along time dim (when enabled) and then drop the prepended part.
        x0 = x0_or_cond
        if self.x0_is_use_noise:
            if x0 is None:
                raise ValueError("FSI mode requires x0 when x0_is_use_noise=True.")
            t0 = x0.shape[1]
            x_in = torch.cat([x0, x], dim=1)
            out = self._forward_impl(x_in, t)
            return out[:, t0:]

        # If not using x0, behave like a plain conditional model.
        return self._forward_impl(x, t)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNO3d(in_dim  = 4,              # Number of input channels.
                in_size = 64,                # Input spatial size
                N_layers = 2,                # Number of (D) and (U) Blocks in the network
                out_dim = 1,                 # Output channels
                ).to(device)
    
    # print(model)
    batch_size = 3
    x = torch.randn(batch_size, 16, 64, 20, 4).to(device)  # (B, T, H, W, C)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

