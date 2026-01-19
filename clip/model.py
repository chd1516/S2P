from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class TeLU(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh( torch.exp(input) )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.last_attn = None  

    def attention(self, x):
        attn_output, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
            attn_mask=self.attn_mask
        )
        self.last_attn = attn_weights.detach()
        return attn_output

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




class SharedSelectivePrompt(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, shared_dim: int, 
                 depth: int, shared_layers=4, hidden_dim=64):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim
        self.depth = depth
        self.shared_layers = shared_layers
        
        self.vision_proj_in = nn.Linear(vision_dim, shared_dim)
        self.text_proj_in = nn.Linear(text_dim, shared_dim)
        self.vision_proj_out = nn.Linear(shared_dim, vision_dim)
        self.text_proj_out = nn.Linear(shared_dim, text_dim)
        

        self.prompt_scale = nn.Parameter(torch.zeros(depth, shared_dim))
        self.prompt_shared_scale = nn.Parameter(torch.zeros(depth // shared_layers, shared_dim))
        
        self.prompt_generator_shared = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                nn.Linear(hidden_dim, shared_dim),
                nn.SiLU()
            ) for _ in range(depth // shared_layers)
        ])

        
        self.prompt_generator_vision = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                nn.Linear(hidden_dim, shared_dim),
                nn.SiLU()
            ) for _ in range(depth)
        ])
        
        self.prompt_generator_text = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim),
                nn.Linear(hidden_dim, shared_dim),
                nn.SiLU()
            ) for _ in range(depth)
        ])

        self.gate_modules_vision = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(depth)
        ])
        
        self.gate_modules_text = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(depth)
        ])
        
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.vision_proj_in.weight)
        nn.init.xavier_uniform_(self.text_proj_in.weight)
        nn.init.xavier_uniform_(self.vision_proj_out.weight)
        nn.init.xavier_uniform_(self.text_proj_out.weight)
        nn.init.zeros_(self.vision_proj_in.bias)
        nn.init.zeros_(self.text_proj_in.bias)
        nn.init.zeros_(self.vision_proj_out.bias)
        nn.init.zeros_(self.text_proj_out.bias)
    
        nn.init.normal_(self.prompt_scale, mean=0.0, std=0.02)
        nn.init.normal_(self.prompt_shared_scale, mean=0.0, std=0.02)

    
        def init_prompt_generator(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for m in self.prompt_generator_shared:
            m.apply(init_prompt_generator)

        for m in self.prompt_generator_vision:
            m.apply(init_prompt_generator)
        for m in self.prompt_generator_text:
            m.apply(init_prompt_generator)

        for gate_module in self.gate_modules_vision + self.gate_modules_text:
            for layer in gate_module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            
            gate_module[-2].bias.data.fill_(-2.0)
            
    def forward(self, x: torch.Tensor, layer_idx: int, modality: str):
        if modality == 'vision':
            x_proj = self.vision_proj_in(x)
        else:  # text
            x_proj = self.text_proj_in(x)
        
       
        prompt_ind = layer_idx // self.shared_layers
        shared_prompt = self.prompt_generator_shared[prompt_ind](x_proj)
        
        if modality == 'vision':
            deep_prompt = self.prompt_generator_vision[layer_idx](x_proj)
            gate = self.gate_modules_vision[layer_idx](x_proj)  # gate
        else:  # text
            deep_prompt = self.prompt_generator_text[layer_idx](x_proj)
            gate = self.gate_modules_text[layer_idx](x_proj) #gate

        
        prompt = (
            self.dropout(shared_prompt) * self.prompt_shared_scale[prompt_ind]+
            self.dropout(deep_prompt) * self.prompt_scale[layer_idx]
        )
        
        
        prompt = prompt * gate 
        prompt = prompt
        
        
        if modality == 'vision':
            prompt = self.vision_proj_out(prompt)
        else:  # text
            prompt = self.text_proj_out(prompt)
        
        return x + prompt       


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 share_dim: int, 
                 hidden_dim: int,
                 layer: int,
                 share_layer: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.layer = layer

        # modified by @lerogo
        self.embed_dim = embed_dim
        

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        for idx, block in enumerate(self.visual.transformer.resblocks):
            for p in block.parameters():
                p.requires_grad = False
        for idx, block in enumerate(self.transformer.resblocks):
            for p in block.parameters():
                p.requires_grad = False

        # 创建共享的选择性提示模块
        self.shared_prompt = SharedSelectivePrompt(
            vision_dim=vision_width,
            text_dim=transformer_width,
            shared_dim=share_dim, 
            depth=max(vision_layers, transformer_layers),  
            shared_layers=share_layer,
            hidden_dim=hidden_dim
        )
        
        self.vision_layers = vision_layers
        self.text_layers = transformer_layers
        
        self.freeze_original_weights()
        

        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.initialize_parameters()
        
    def freeze_original_weights(self):
        
        for name, param in self.named_parameters():
            if "shared_prompt" not in name:
                param.requires_grad = False
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        if isinstance(self.visual, VisionTransformer):
            x = self.visual.conv1(image.type(self.dtype))  # [B, C, H, W]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N, D]
    
            cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # LND
            hidden_states = x

            for idx, block in enumerate(self.visual.transformer.resblocks):
                if idx == self.layer or self.layer == 100:
                    # Selective Prompting
                    hidden_states = hidden_states.permute(1, 0, 2)  # BLD
                    hidden_states = self.shared_prompt(hidden_states, layer_idx=idx, modality='vision')
                    hidden_states = hidden_states.permute(1, 0, 2)  # LBD
                    hidden_states = block(hidden_states)
                else:
                    
                    hidden_states = block(hidden_states)


            x = hidden_states.permute(1, 0, 2)  # [B, N, D]
            x_3 = self.visual.ln_post(x)
            x_cls = x_3[:, 0, :]
            x_cls = x_cls @ self.visual.proj
            x_3 = x_3 @ self.visual.proj
            return x_cls
        else:
            return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [B, L, D]
        x = x + self.positional_embedding[:x.size(1)].type(self.dtype)  # ensure alignment with context_length
        x = x.permute(1, 0, 2)  # [L, B, D]

        hidden_states = x
        
        for idx, block in enumerate(self.transformer.resblocks):
        
            if idx == self.layer or self.layer == 100:
                # Add SVP prompt before each transformer layer
                hidden_states = hidden_states.permute(1, 0, 2)  # [B, L, D]
                hidden_states = self.shared_prompt(hidden_states, layer_idx=idx, modality='text')
                hidden_states = hidden_states.permute(1, 0, 2)  # [L, B, D]
                hidden_states = block(hidden_states)
            else:
                hidden_states = block(hidden_states)


        x = hidden_states.permute(1, 0, 2)  # [B, L, D]
        x_3 = self.ln_final(x).type(self.dtype)
    
        x_cls = x_3[torch.arange(x_3.shape[0]), text.argmax(dim=-1)]  # EOT token
        x_proj = x_cls @ self.text_projection
        x_3 = x_3 @ self.text_projection
        return x_proj

    def forward(self, image, text):

        return image_features, text_features


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def convert_models_to_fp32(model):
    """Convert applicable model parameters to fp32"""
    # modified by @lerogo
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()


def build_model(state_dict: dict, share_dim, hidden_dim,layer, share_layer):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, share_dim, hidden_dim,layer,share_layer
    )

    # modified by @lerogo
    # for key in ["input_resolution", "context_length", "vocab_size"]:
    for key in ["input_resolution", "context_length", "vocab_size", "logit_scale"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    return model.eval()
