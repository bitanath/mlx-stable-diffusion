import math
import mlx.core as mx
import mlx.nn as nn

##TODO: Ensure all Group Norms are pytorch_compatible=True for one to one correspondence
##NOTE: All inputs and outputs from Attention and residual are NHWC and NOT NCHW as in pytorch

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def __call__(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        qkv = self.in_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1) # From Pytorch torch.chunk to MLX.split
        
        q = mx.reshape(q, interim_shape)
        q = mx.transpose(q, (0, 2, 1, 3))  # (batch, n_heads, seq_len, d_head)
        k = mx.reshape(k, interim_shape)
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.reshape(v, interim_shape)
        v = mx.transpose(v, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 1, 3, 2))  # Transpose last two dimensions like the torch impl
        weight = mx.matmul(q, k_t)
        
        if causal_mask:
            mask = mx.triu(mx.ones((sequence_length, sequence_length)), k=1)
            mask = mx.expand_dims(mx.expand_dims(mask, 0), 0)  # Add batch and head dims
            weight = mx.where(mask, mx.full(weight.shape, -float('inf')), weight)
        
        weight = weight / math.sqrt(self.d_head)
        weight = mx.softmax(weight, axis=-1)
        
        output = mx.matmul(weight, v)
        
        output = mx.transpose(output, (0, 2, 1, 3))  # (batch, seq_len, n_heads, d_head)
        output = mx.reshape(output, input_shape)
        
        output = self.out_proj(output)
        
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def __call__(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = mx.reshape(q, interim_shape)
        q = mx.transpose(q, (0, 2, 1, 3))  # (batch, n_heads, seq_len, d_head)
        k = mx.reshape(k, interim_shape)
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.reshape(v, interim_shape)
        v = mx.transpose(v, (0, 2, 1, 3))
        
        k_t = mx.transpose(k, (0, 1, 3, 2))  # Transpose last two dimensions
        weight = mx.matmul(q, k_t)
        
        weight = weight / math.sqrt(self.d_head)
        weight = mx.softmax(weight, axis=-1)
        
        output = mx.matmul(weight, v)
        
        output = mx.transpose(output, (0, 2, 1, 3))  # (batch, seq_len, n_heads, d_head)
        output = mx.reshape(output, input_shape)
        output = self.out_proj(output)
        
        return output

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = mx.zeros((n_token, n_embd))
    
    def __call__(self, tokens):
        x = self.token_embedding(tokens)
        y = x + self.position_embedding
        
        return y

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x = x + residue  
        
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        
        x = x * mx.sigmoid(1.702 * x) # Fast GELU approximation: x * sigmoid(1.702 * x)
        x = self.linear_2(x)
        y = x + residue  

        return y

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = [CLIPLayer(12, 768) for i in range(12)]

        self.layernorm = nn.LayerNorm(768)
    
    def __call__(self, tokens):
        
        tokens = tokens.astype(mx.int32) #NOTE: MLX does not have a long type, so using int32
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
    

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def __call__(self, x):
        x = self.linear_1(x)
        x = nn.silu(x) 
        x = self.linear_2(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def __call__(self, x):
        # For NHWC format, shape is (batch, height, width, channels)
        batch, height, width, channels = x.shape
        x_reshaped = mx.reshape(x, (batch, height, 1, width, 1, channels))
        x_repeated = mx.tile(x_reshaped, (1, 1, 2, 1, 2, 1))
        x_upsampled = mx.reshape(x_repeated, (batch, height * 2, width * 2, channels))
        
        return self.conv(x_upsampled)
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels, pytorch_compatible=True)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels, pytorch_compatible=True)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def __call__(self, feature:mx.array, time:mx.array):
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = nn.silu(feature)
        feature = self.conv_feature(feature)
        time = nn.silu(time)
        time = self.linear_time(time)

        time_expanded = mx.expand_dims(mx.expand_dims(time, 1), 1) #TODO: Expand from N,C to N,H,W,C
        merged = feature + time_expanded
        merged = self.groupnorm_merged(merged)
        merged = nn.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
    

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def __call__(self, x:mx.array, context:mx.array):
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, h, w, c = x.shape
        x = x.reshape((n, h * w, c))
        residue_short = x
    
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        residue_short = x

        x = self.layernorm_3(x)
        x, gate = mx.split(self.linear_geglu_1(x),2,axis=-1) 
        x = x * nn.gelu(gate) #TODO: Can this be replaced with fast approx gelu?
        
        x = self.linear_geglu_2(x)
        
        y = x + residue_short
        y = y.reshape((n, h, w, c)) #NOTE: Differs from pytorch which is nchw

        return self.conv_output(y) + residue_long
    

class SwitchSequential:
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def __call__(self, x:mx.array, context:mx.array=None, time:mx.array=None):
        for layer in self.layers:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
    
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = [
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)), #8ths (H//8,W//8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), #8ths
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), #8ths
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)), #8ths
        
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)), #16ths
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)), #16ths
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)), #32ths
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)), #32ths
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)), #32ths
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)), #64ths
            SwitchSequential(UNET_ResidualBlock(1280, 1280)), #64ths
            SwitchSequential(UNET_ResidualBlock(1280, 1280)), #64ths
        ]

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), #64ths 
            UNET_AttentionBlock(8, 160),  #64ths
            UNET_ResidualBlock(1280, 1280), #64ths
        )
        
        self.decoders = [
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), #64ths
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), #64ths
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)), #32ths
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),# 32ths
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)), # 32ths
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)), #16ths
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)), #16ths
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)), #16ths
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)), #8ths
            
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)), #8ths
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), #8ths
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), #8ths
        ]

    def __call__(self, x:mx.array, context:mx.array, time:mx.array):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
            
        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = mx.concatenate([x, skip_connections.pop()], axis=-1)  #NOTE: axis=-1 for NHWC differs from pytorch NCHW
            x = layers(x, context, time)
        
        return x
    
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels,pytorch_compatible=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def __call__(self, x):
        x = self.groupnorm(x)
        x = nn.silu(x)
        x = self.conv(x)
        return x
    

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def __call__(self, latent, context, time):
        time = self.time_embedding(time)  # (1, 320) -> (1, 1280)
        output = self.unet(latent, context, time)  # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.final(output)  # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8) to be decoded with VAE
        return output
    

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def __call__(self, x):
        return mx.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = [
            conv(n_in, n_out), 
            nn.ReLU(), 
            conv(n_out, n_out), 
            nn.ReLU(), 
            conv(n_out, n_out)
        ]
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        
    def __call__(self, x):
        conv_out = x
        for layer in self.conv:
            conv_out = layer(conv_out)
        
        return self.fuse(conv_out + self.skip(x))

def TinyEncoder(latent_channels=4):
    layers = [
        conv(3, 64), 
        Block(64, 64),
        conv(64, 64, stride=2, bias=False), 
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64),
        conv(64, 64, stride=2, bias=False), 
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64),
        conv(64, 64, stride=2, bias=False), 
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64),
        conv(64, latent_channels),
    ]
    
    def forward(x):
        for layer in layers:
            x = layer(x)
        return x
    return layers, forward

def TinyDecoder(latent_channels=4):
    layers = [
        Clamp(), 
        conv(latent_channels, 64), 
        nn.ReLU(),
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64), 
        lambda x: mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2),  # nhwc -> n,h*2,w*2,c
        conv(64, 64, bias=False),
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64), 
        lambda x: mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2),  # nhwc -> n,h*2,w*2,c
        conv(64, 64, bias=False),
        Block(64, 64), 
        Block(64, 64), 
        Block(64, 64), 
        lambda x: mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2),  # nhwc -> n,h*2,w*2,c
        conv(64, 64, bias=False),
        Block(64, 64), 
        conv(64, 3),
    ]
    
    def forward(x):
        for layer in layers:
            x = layer(x)
        return x
    
    return layers, forward

class Encoder(nn.Module):
    def __init__(self, encoder_layers=None, encoder_fn=None):
        super().__init__()
        if encoder_layers is None or encoder_fn is None:
            self.encoder_layers, self.encoder_fn = TinyEncoder()
        else:
            self.encoder_layers = encoder_layers
            self.encoder_fn = encoder_fn

    def __call__(self, x):
        x = mx.clip(x, 0, 1)
        return self.encoder_fn(x)
    
class Decoder(nn.Module):
    def __init__(self, decoder_layers=None, decoder_fn=None):
        super().__init__()
        if decoder_layers is None or decoder_fn is None:
            self.decoder_layers, self.decoder_fn = TinyDecoder()
        else:
            self.decoder_layers = decoder_layers
            self.decoder_fn = decoder_fn

    def __call__(self, x):
        output = self.decoder_fn(x)
        return mx.clip(output, 0, 1)