import torch
import torch.nn as nn
import math
from functools import reduce

def setup_huggingface_auth():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not found. Authentication may fail.")
        return False
    login(token=hf_token)
    return True

def get_module_by_name(model, access_string):
    names = access_string.split('.')
    return reduce(getattr, names, model)

class LoRAMultiheadAttention(nn.Module):
    def __init__(self, original_mha, rank=4, alpha=16):
        super().__init__()
        self.original_mha = original_mha
        device = next(original_mha.parameters()).device
        
        # Freeze the original attention module
        for param in self.original_mha.parameters():
            param.requires_grad = False
        
        embed_dim = self.original_mha.embed_dim
        
        # Create LoRA parameters for input projections (q, k, v)
        self.q_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.q_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.k_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.k_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        self.v_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.v_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        # Create LoRA parameters for output projection
        self.out_lora_A = nn.Parameter(torch.zeros((rank, embed_dim), device=device))
        self.out_lora_B = nn.Parameter(torch.zeros((embed_dim, rank), device=device))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_lora_B)
        nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.k_lora_B)
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_lora_B)
        nn.init.kaiming_uniform_(self.out_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.out_lora_B)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        device = query.device
        
        # Ensure LoRA parameters are on the same device
        if self.q_lora_A.device != device:
            self.q_lora_A = self.q_lora_A.to(device)
            self.q_lora_B = self.q_lora_B.to(device)
            self.k_lora_A = self.k_lora_A.to(device)
            self.k_lora_B = self.k_lora_B.to(device)
            self.v_lora_A = self.v_lora_A.to(device)
            self.v_lora_B = self.v_lora_B.to(device)
            self.out_lora_A = self.out_lora_A.to(device)
            self.out_lora_B = self.out_lora_B.to(device)
        
        attn_output, attn_weights = self.original_mha(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=need_weights, 
            attn_mask=attn_mask
        )
        
        # LoRA contributions
        q_lora = query @ self.q_lora_A.T @ self.q_lora_B.T * self.scaling
        k_lora = key @ self.k_lora_A.T @ self.k_lora_B.T * self.scaling
        v_lora = value @ self.v_lora_A.T @ self.v_lora_B.T * self.scaling
        
        lora_contribution = q_lora + k_lora + v_lora
        attn_output = attn_output + lora_contribution
        
        return attn_output, attn_weights

def apply_lora_to_multihead_attention(model, rank=4, alpha=16):
    """Replace MultiheadAttention modules with LoRA-enabled versions"""
    replaced_modules = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            print(f"Found MultiheadAttention module: {name}")
            
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                try:
                    parent_module = get_module_by_name(model, parent_name)
                    lora_attention = LoRAMultiheadAttention(module, rank=rank, alpha=alpha)
                    lora_attention = lora_attention.to(device)
                    setattr(parent_module, child_name, lora_attention)
                    replaced_modules[name] = lora_attention
                    print(f"Applied LoRA to {name}")
                except Exception as e:
                    print(f"Error applying LoRA to {name}: {e}")
    
    print(f"Applied LoRA to {len(replaced_modules)} attention modules")
    return model, replaced_modules
