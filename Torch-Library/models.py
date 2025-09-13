#!# Much of this model's code is based off of CLIP https://github.com/openai/CLIP/tree/main
### Aim is to make a minimal VLM and then re-implement it from scratch in Pytorch in Torch-Scratch directory
import torch
from torch import nn
import torchvision
import numpy as np



#!# Wrapper classes of nn modules, to be coded from scratch in Torch-Scratch directory version

### Need layernorm 
class Linear_Custom(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
    def forward(self, x):
        return super().forward(x)


### Need layernorm 
class Conv2d_Custom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, bias=bias)
    def forward(self, x):
        return super().forward(x)


### Need to implement MHA layer
class MultiHead_Attention_Custom(nn.MultiheadAttention):
    def __init__(self, d_in, n_heads):
        super().__init__(d_in, n_heads)
    def forward(self, q, k, v, need_weights, attn_mask):
        return super().forward(q, k, v, need_weights=need_weights, attn_mask=attn_mask)


### Need layernorm 
class LayerNorm_Custom(nn.LayerNorm):
    def __init__(self, width):
        super().__init__(width)
    def forward(self, x):
        return super().forward(x)











### Need linear layer 
class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_up = Linear_Custom(width, width * 2)
        ### Using ReLU over GELU for practice since its going to be easier to remember how to implement from scratch
        self.activation = nn.ReLU()
        self.linear_down = Linear_Custom(width * 2, width)

    def forward(self, x):
        x = self.activation(self.linear_up(x))
        x = self.linear_down(x)
        return x



### Need a residual block
class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, n_heads, attn_mask=None):    
        super().__init__()
        self.attn_mask = attn_mask
        self.mha = MultiHead_Attention_Custom(width, n_heads)
        self.ln_1 = LayerNorm_Custom(width)

        self.ff = FeedForward(width)
        self.ln_2 = LayerNorm_Custom(width)

    #!# Masks for text branch but not visual (mask is None)
    def masked_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.mha(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    ### Using the pre-norm setup
    def forward(self, x):
        x = x + self.masked_attention(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


### Wanted to try handle embedding differently from using a wrapper around Transformer_Branch like CLIP would
class Image_Embedder(nn.Module):
    def __init__(self, img_size, patch_size, width, output_dim, layers=1):
        super().__init__()
        self.conv = Conv2d_Custom(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.n_patches = (img_size // patch_size) ** 2
        self.n_filters = width
        self.ln_pre = LayerNorm_Custom(width)
        self.ln_post = LayerNorm_Custom(width)

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, width//64) for _ in range(layers)])

        scale = width ** -0.5
        self.class_embed = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(scale * torch.randn(self.n_patches + 1, width))
        self.project = nn.Parameter(scale * torch.randn(width, output_dim))



    def forward(self,x):
        x = self.conv(x)
        
        x = x.reshape(x.shape[0], self.n_filters, -1)  ### Flatten feature maps
        x = x.permute(0,2,1)                             ### Swap n_filters and n_patches
        x = torch.cat([self.class_embed.to(x.dtype) + torch.zeros(x.shape[0], 1, self.n_filters, dtype=x.dtype, device=x.device), 
                        x], dim=1)
        x = x + self.pos_embed.to(x.dtype)
        ### Pre-normalize so residual is normalized in first resblock
        x = self.ln_pre(x)


        x = x.permute(1,0,2) ### NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1,0,2) ### LND -> NLD 

        ### Normalize the class embedding token
        x = self.ln_post(x[:,0,:])

        ### Project from n_filters width to embedding dimension
        x = x @ self.project

        return x






#!# Making dedicated text embedder class to make the code more symmetric between modalities
class Text_Embedder(nn.Module):
    def __init__(self, vocab_size, context_length, width, output_dim, attn_mask, dtype, layers=1):
        super().__init__()
        self.context_length = context_length
        self.weight_type = dtype
        
        self.ln_pre = LayerNorm_Custom(width)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, width//64, attn_mask) for _ in range(layers)])
        self.ln_post = LayerNorm_Custom(width)
        
        self.pos_embed = nn.Parameter(torch.empty(self.context_length, width))
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.project = nn.Parameter(torch.empty(width, output_dim))

        self.initialize_parameters(width)


    def forward(self, text):
        x = self.token_embedding(text).type(self.weight_type)  # [batch_size, n_ctx, d_model]
        x = x + self.pos_embed

        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        ### Normalize the class embedding token
        x = self.ln_post(x).type(self.weight_type) #[batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.project

        return x



    def initialize_parameters(self, width):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        if self.project is not None:
            nn.init.normal_(self.project, std=width**-0.5)


        proj_std = (width**-0.5) * ((2 * width)**-0.5)
        attn_std = width**-0.5
        fc_std = (2 * width)**-0.5
        for block in self.resblocks:
            nn.init.normal_(block.mha.in_proj_weight, std=attn_std)
            nn.init.normal_(block.mha.out_proj.weight, std=proj_std)
            nn.init.normal_(block.ff.linear_up.weight, std=fc_std)
            nn.init.normal_(block.ff.linear_down.weight, std=proj_std)






class Minimal_VLM(nn.Module):
    def __init__(self, 
                output_dim,
                img_size, patch_size, vision_width,            # Vision 
                vocab_size, context_length, transformer_width  # Text
            ):
        super().__init__()
        
        self.context_length = context_length
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer_width = transformer_width
    
        self.vision_transformer = Image_Embedder(
                                    img_size = img_size, 
                                    patch_size = patch_size, 
                                    width = vision_width, 
                                    output_dim = output_dim
                                )
                                
        self.weight_type = self.vision_transformer.conv.weight.dtype

        self.text_transformer = Text_Embedder(
                                    vocab_size = vocab_size, 
                                    context_length = context_length, 
                                    width = transformer_width,
                                    output_dim = output_dim, 
                                    attn_mask = self.build_attention_mask(),
                                    dtype=self.weight_type
                                )



    ### Need overall VLM functions for forward() and initialization
    def forward(self, images, text):
        image_features = self.vision_transformer(images.type(self.weight_type))
        text_features = self.text_transformer(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text









    def build_attention_mask(self):
        # lazily create causal attention mask for text data
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask



