# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, repeat
import pdb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                               FNO Components                                  #
#################################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 modes1, 
                 modes2, 
                 modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, \
                                    self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, \
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = \
            self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:] = \
            self.compl_mul1d(x_ft[:, :, -self.modes1:], self.weights2)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CompactFNOFinalLayer(nn.Module):
    """
    A compact 1D FNO-based final layer for SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, modes=16):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Handle rectangular patch sizes
        if isinstance(patch_size, (tuple, list)):
            patch_area = patch_size[0] * patch_size[1]
        else:
            patch_area = patch_size * patch_size
            
        self.out_channels = out_channels
        self.patch_area = patch_area
        
        # Compact 1D FNO components
        self.modes = modes
        
        # Input projection to FNO width
        self.fno_width = min(64, hidden_size // 2)  # Keep FNO compact
        self.input_proj = nn.Linear(hidden_size, self.fno_width)
        
        # Single 1D FNO layer (compact) - operates along the patch dimension
        self.spectral_conv = SpectralConv1d(self.fno_width, self.fno_width, self.modes)
        self.conv = nn.Conv1d(self.fno_width, self.fno_width, 1)
        self.bn = nn.BatchNorm1d(self.fno_width)
        
        # Output projection
        self.output_proj = nn.Linear(self.fno_width, patch_area * out_channels, bias=True)
        
        # Adaptive layer norm modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (N, num_patches, hidden_size) - patch tokens
        c: (N, hidden_size) - conditioning
        """
        # Apply adaptive layer norm modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        
        # Project to FNO width
        x = self.input_proj(x)  # (N, num_patches, fno_width)
        
        # Transpose for 1D convolution: (N, fno_width, num_patches)
        x = x.transpose(1, 2)  # (N, fno_width, num_patches)
        
        # Apply 1D FNO layer along the patch dimension
        x1 = self.spectral_conv(x)  # (N, fno_width, num_patches)
        x2 = self.conv(x)           # (N, fno_width, num_patches)
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        
        # Transpose back: (N, num_patches, fno_width)
        x = x.transpose(1, 2)
        
        # Project to output
        x = self.output_proj(x)  # (N, num_patches, patch_area * out_channels)
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Handle rectangular patch sizes
        if isinstance(patch_size, (tuple, list)):
            patch_area = patch_size[0] * patch_size[1]
        else:
            patch_area = patch_size * patch_size
        self.linear = nn.Linear(hidden_size, patch_area * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT_FNO(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(64, 64),  # Changed to tuple for (height, width)
        patch_size=(2, 2),
        in_channels=4,
        out_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        x0_is_use_noise=True,
        dataset_name="turek_hron_data",
        stage='fluid', # fluid, structure
        forward_type='latte',
        use_surrogate=False,
        modes=16,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.stage = stage
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dataset_name = dataset_name
        self.forward_type = forward_type
        
        self.use_surrogate = True # use_surrogate
        
        # Handle input_size as either tuple or single value for backward compatibility
        if isinstance(input_size, (int, float)):
            input_height = input_width = int(input_size)
        else:
            input_height, input_width = input_size
            
        # if x0_is_use_noise:
        #     self.x_embedder = PatchEmbed(input_size, patch_size, in_channels*2, hidden_size, bias=True)
        # else:
        #     self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        print("input_size:", input_size)
        print("patch_size:", patch_size)
        print("in_channels:", in_channels)
        
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        print("num_patches:", num_patches)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.final_layer = CompactFNOFinalLayer(hidden_size, patch_size, out_channels=self.out_channels, modes=modes)
        self.x0_is_use_noise = x0_is_use_noise
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # For rectangular images, use the actual grid dimensions
        grid_height = self.x_embedder.grid_size[0]
        grid_width = self.x_embedder.grid_size[1]
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_height, grid_width)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize temporal embedding:
        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)
        
        nn.init.constant_(self.final_layer.output_proj.weight, 0)
        nn.init.constant_(self.final_layer.output_proj.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p_h, p_w = self.x_embedder.patch_size
        h = self.x_embedder.grid_size[0]
        w = self.x_embedder.grid_size[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p_h, w * p_w))
        return imgs

    def data_preprocess(self, xt, x0):
        
        """
        Improved Latte forward pass based on official implementation.
        xt: (N, T1, H, W, C) - target frames to denoise
        x0: (N, T2, H, W, C) - initial frames  
        t: (N,) - diffusion timesteps
        cond: additional conditioning information
        """
        # Convert to (N, T, C, H, W) format
        xt = xt.permute(0,1,4,2,3).float()
        x0 = x0.permute(0,1,4,2,3).float()
        
        T1 = xt.shape[1]
        T2 = x0.shape[1]
        
        # pdb.set_trace()
        if self.x0_is_use_noise:
            x = torch.cat([x0, xt], dim=1)
        elif self.use_surrogate:
            x = torch.cat([x0, xt], dim=1)
        else:
            x = xt
            
        return x, T2
        
            
    def data_postprocess(self, x, T2=None):
        if self.use_surrogate:
            x = x[:, T2:].permute(0, 1, 3, 4, 2)
        else:
            x = x[:, T2:].permute(0, 1, 3, 4, 2)

        return x

    def forward_SiT(self, xt, t, x0, cond):
        """
        Forward pass of SiT.
        xt: (N, C, H, W) 
        t: (N,) tensor of diffusion timesteps
        x0: (N, C, H, W) 
        channel: pressure, x_velocity, y_velocity, sdf
        """
        # def forward(self, xt, t, x0, cond):
        xt, x0 = self.apply_mask(xt, x0)
        
        # (N, 1, H, W, C) 
        xt=xt.squeeze(1).permute(0,3,1,2).float()
        x0=x0.squeeze(1).permute(0,3,1,2).float()
        
        # print("x0_is_use_noise:", self.x0_is_use_noise)
        if self.x0_is_use_noise:
            x = torch.cat([xt, x0], dim=1)
        else:
            x = xt
            
        # print(x.shape)
        # print(self.x_embedder(x).shape)
        # print(self.pos_embed.shape)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        
        # if 'reynolds_number' in cond:
        #     c = t + (cond['reynolds_number']/10000).to(t.device)
        # elif 're' in cond:
        #     c = t + (cond['re']/10000).to(t.device)
        # else:
        #     c = t
        c = t
        
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        
        x = x.permute(0,2,3,1).unsqueeze(1)
        
        x, _ = self.apply_mask(x)
        
        return x

    def forward_latte(self, xt, t, x0, cond):
        # for k in xt.keys():
        #     print(k, xt[k].shape)
        # pdb.set_trace()
        x, T2 = self.data_preprocess(xt, x0)
            
        batches, frames, channels, height, width = x.shape
        
        # Dynamically adjust temporal embedding to match actual frame count
        if frames != self.temp_embed.shape[1]:
            temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], frames)
            temp_embed = torch.from_numpy(temp_embed).float().unsqueeze(0).to(x.device)
        else:
            temp_embed = self.temp_embed
            
        # Reshape for spatial processing: (b*f, c, h, w)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        
        # Spatial patch embedding + positional encoding
        x = self.x_embedder(x) + self.pos_embed  # (b*f, num_patches, hidden_dim)
        # Timestep embedding
        t_emb = self.t_embedder(t)  # (b, hidden_dim)
        
        # Prepare conditioning embeddings for spatial and temporal blocks
        timestep_spatial = repeat(t_emb, 'b d -> (b f) d', f=frames)  # (b*f, hidden_dim)
        timestep_temp = repeat(t_emb, 'b d -> (b p) d', p=self.pos_embed.shape[1])  # (b*num_patches, hidden_dim)
        
        # Alternating spatial-temporal transformer blocks
        for i in range(0, len(self.blocks), 2):
            # Ensure sufficient block pairs
            if i + 1 >= len(self.blocks):
                # If odd number of blocks, treat the last one as spatial block
                spatial_block = self.blocks[i]
                x = spatial_block(x, timestep_spatial)
                break
                
            spatial_block = self.blocks[i]
            temp_block = self.blocks[i + 1]
            # spatial_block, temp_block = self.blocks[i:i+2]

            c = timestep_spatial
            # Spatial attention block
            x = spatial_block(x, c)
            
            # Reshape for temporal processing: (b*num_patches, frames, hidden_dim)
            x = rearrange(x, '(b f) p d -> (b p) f d', b=batches)
            
            # Add temporal positional embedding (only at first temporal block)
            if i == 0:
                x = x + temp_embed
            
            # Temporal attention block
            c = timestep_temp
            x = temp_block(x, c)
            
            # Reshape back for next spatial block: (b*f, num_patches, hidden_dim)
            x = rearrange(x, '(b p) f d -> (b f) p d', b=batches)
        
        # Final layer with spatial conditioning
        x = self.final_layer(x, timestep_spatial)
        # Unpatchify: (b*f, c, h, w)
        x = self.unpatchify(x)
        
        # Reshape back to video format: (b, f, c, h, w)
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)

        # pdb.set_trace()
        # Apply learned sigma if enabled
        # if self.learn_sigma:
        #     x, _ = x.chunk(2, dim=2)  # split on channel dim
        
        # # Extract target frames if concatenated conditioning was used
        # pdb.set_trace()
        # if self.x0_is_use_noise:
        #     x = x[:, T2:].permute(0, 1, 3, 4, 2)
        
        x = self.data_postprocess(x, T2)
        
        # Convert back to (N, T, H, W, C) format
        return x


    def forward(self, x, t, cond=None, **kwargs):
        """
        Forward pass for SiT FNO.
        x: (B, T, H, W, C) input tensor
        t: (B,) diffusion timestep
        cond: Optional conditioning tensor (currently unused, included for API compatibility)
        """
        # Standardize input shape from (B, T, H, W, C) to (B, C, F, H, W) for the model
        # Note: Model expects 'F' (frames) as 3rd dim, which corresponds to 'T' in input
        x = x.permute(0, 4, 1, 2, 3) # (B, T, H, W, C) -> (B, C, T, H, W)
        
        # Call latte forward logic directly
        # We skip the data_preprocess step that was prepending x0
        
        batches, channels, frames, height, width = x.shape
        
        # Dynamically adjust temporal embedding to match actual frame count
        if frames != self.temp_embed.shape[1]:
            temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], frames)
            temp_embed = torch.from_numpy(temp_embed).float().unsqueeze(0).to(x.device)
        else:
            temp_embed = self.temp_embed
            
        # Reshape for spatial processing: (b*f, c, h, w)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        
        # Spatial patch embedding + positional encoding
        x = self.x_embedder(x) + self.pos_embed  # (b*f, num_patches, hidden_dim)
        
        # Timestep embedding
        t_emb = self.t_embedder(t)  # (b, hidden_dim)
        
        # Prepare conditioning embeddings for spatial and temporal blocks
        timestep_spatial = repeat(t_emb, 'b d -> (b f) d', f=frames)  # (b*f, hidden_dim)
        timestep_temp = repeat(t_emb, 'b d -> (b p) d', p=self.pos_embed.shape[1])  # (b*num_patches, hidden_dim)
        
        # Alternating spatial-temporal transformer blocks
        for i in range(0, len(self.blocks), 2):
            # Ensure sufficient block pairs
            if i + 1 >= len(self.blocks):
                # If odd number of blocks, treat the last one as spatial block
                spatial_block = self.blocks[i]
                x = spatial_block(x, timestep_spatial)
                break
                
            spatial_block = self.blocks[i]
            temp_block = self.blocks[i + 1]
            
            c = timestep_spatial
            # Spatial attention block
            x = spatial_block(x, c)
            
            # Reshape for temporal processing: (b*num_patches, frames, hidden_dim)
            x = rearrange(x, '(b f) p d -> (b p) f d', b=batches)
            
            # Add temporal positional embedding (only at first temporal block)
            if i == 0:
                x = x + temp_embed
            
            # Temporal attention block
            c = timestep_temp
            x = temp_block(x, c)
            
            # Reshape back for next spatial block: (b*f, num_patches, hidden_dim)
            x = rearrange(x, '(b p) f d -> (b f) p d', b=batches)
        
        # Final layer with spatial conditioning
        x = self.final_layer(x, timestep_spatial)
        # Unpatchify: (b*f, c, h, w)
        x = self.unpatchify(x)
        
        # Reshape back to video format: (b, c, f, h, w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batches)
        
        # Permute back to (B, T, H, W, C) to match expected output format in train.py
        x = x.permute(0, 2, 3, 4, 1) 

        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])

    def apply_mask(self, xt, x0=None):
        if self.stage == 'fluid':
            xt_mask = torch.ones_like(xt)
            # xt_mask[..., -1:] = 0 # reserve u_noise, v_noise, pressure_noise
            xt = xt * xt_mask 
            
            if x0 is not None:
                x0_mask = torch.ones_like(x0)
                # x0_mask[..., -1:] = 0 # u, v, sdf, pressure
                x0 = x0 * x0_mask
            
        elif self.stage == 'structure':
            xt_mask = torch.ones_like(xt)
            # xt_mask[..., :-1] = 0 # reserve sdf_noise
            xt = xt * xt_mask
            
            if x0 is not None:
                x0_mask = torch.ones_like(x0) 
                # x0_mask[..., :-2] = 0 # reserve sdf, pressure
                x0 = x0 * x0_mask
            
        elif self.stage == 'fsi':
            xt = xt
            if x0 is not None:
                x0 = x0
            
        return xt, x0


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    """
    Generate 1D sinusoidal temporal embeddings.
    """
    pos = np.arange(0, length, dtype=np.float32).reshape(-1, 1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width=None, cls_token=False, extra_tokens=0):
    """
    grid_height, grid_width: int of the grid height and width (can be different for rectangular images)
    return:
    pos_embed: [grid_height*grid_width, embed_dim] or [1+grid_height*grid_width, embed_dim] (w/ or w/o cls_token)
    """
    if grid_width is None:
        grid_width = grid_height  # For backward compatibility with square images
    
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT_FNO(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT_FNO(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT_FNO(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT_FNO(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT_FNO(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT_FNO(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT_FNO(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT_FNO(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT_FNO(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT_FNO(input_size=(64, 128), depth=12, hidden_size=384, patch_size=(2, 4), num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT_FNO(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT_FNO(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

if __name__ == "__main__":
    import torch

    model = SiT_S_2(num_classes=1000, learn_sigma=True)
    
    batch_size = 2
    channels = 4
    image_size = 64
    
    x = torch.randn(batch_size, 1, image_size, image_size*2, channels)
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, 1, image_size, image_size*2, channels)
    
    output = model(x, t, y, cond=None)
    
    print(f"Input shape: {x.shape}")
    print(t.shape)
    print(y.shape)
    print(f"Output shape: {output.shape}")