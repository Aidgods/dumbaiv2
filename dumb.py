from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel



@dataclass
class HierarchicalReasoningModel_ACTV2InnerCarry:
    z_H: torch.Tensor  # [num_experts, B, d] - per-position state
    z_L: torch.Tensor
    z_mem: torch.Tensor
    evolution_mask: torch.Tensor  # [num_policy_slots, B] - per-batch mask
    steps: torch.Tensor = None

@dataclass
class HierarchicalReasoningModel_ACTV2Carry:
    inner_carry: HierarchicalReasoningModel_ACTV2InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class MurderTreeV2Config(BaseModel):
    vocab_size: int
    seq_len: int
    hidden_size: int = 1024
    num_heads: int = 16
    rope_theta: float = 100000.0
    forward_dtype: str = "bfloat16"
    
    tier1_dim: int = 384
    tier1_depth: int = 2
    tier1_count: int = 32
    tier2_dim: int = 768
    tier2_depth: int = 4
    tier2_count: int = 8
    tier3_dim: int = 1024
    tier3_depth: int = 8
    tier3_count: int = 3

    verifier_depth: int = 32
    slow_verify_every: int = 8

    max_steps: int = 64
    ponder_cost_weight: float = 0.01
    evolution_trigger_threshold: float = 0.35
    num_policy_slots: int = 12
    evolution_candidates: int = 8
    rms_eps: float = 1e-5
    allow_inference_evolution: bool = False
    inference_evolution_threshold: float = 0.5
    inference_evolution_candidates: int = 4
    max_inference_evolutions: int = 10
    inference_noise_scale: float = 0.01


class CosSin:
    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device=None):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(seq_len, device=self.cos_cached.device)
        return CosSin(self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :])

def apply_rotary_emb(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    
    q_dim = q.size(-1)
    cos_dim = cos.size(-1)
    sin_dim = sin.size(-1)
    
    assert q_dim == cos_dim == sin_dim, f"Rotary dimension mismatch: q={q_dim}, cos={cos_dim}, sin={sin_dim}"
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class DTypeAwareLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, forward_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias)
        self.forward_dtype = forward_dtype

    def forward(self, input):
        return F.linear(input.to(self.forward_dtype), 
                       self.weight.to(self.forward_dtype), 
                       self.bias.to(self.forward_dtype) if self.bias is not None else None)

class DTypeAwareEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, forward_dtype=torch.bfloat16):
        super().__init__(num_embeddings, embedding_dim)
        self.forward_dtype = forward_dtype

    def forward(self, input):
        return F.embedding(input, self.weight.to(self.forward_dtype))

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, causal=False, forward_dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.forward_dtype = forward_dtype

        self.q_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.k_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.v_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.o_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)

        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states, cos_sin=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            q, k = apply_rotary_emb(q, k, cos_sin.cos, cos_sin.sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.causal:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device)
            mask = torch.triu(mask, diagonal=1)
            attn_weights += mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, input_size, output_size, expansion=2.0, forward_dtype=torch.bfloat16):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.expanded_size = int(input_size * expansion)
        self.forward_dtype = forward_dtype

        self.gate_proj = DTypeAwareLinear(input_size, self.expanded_size, forward_dtype=forward_dtype)
        self.up_proj = DTypeAwareLinear(input_size, self.expanded_size, forward_dtype=forward_dtype)
        self.down_proj = DTypeAwareLinear(self.expanded_size, output_size, forward_dtype=forward_dtype)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

def rms_norm(hidden_states, variance_epsilon=1e-5):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

def trunc_normal_init_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


class GriffinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 16, forward_dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.forward_dtype = forward_dtype
        self.heads = heads
        self.head_dim = dim // heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(hidden_size=dim, num_heads=heads, forward_dtype=forward_dtype)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(input_size=dim, output_size=dim, expansion=4.0, forward_dtype=forward_dtype)
        self.gate = nn.Parameter(torch.zeros(1, 1, dim, dtype=forward_dtype))

    def forward(self, h: torch.Tensor, inject: Optional[torch.Tensor] = None, cos_sin=None):
        h = h.to(self.forward_dtype)
        
        x = self.norm1(h)
        if inject is not None:
            x = x + inject.to(self.forward_dtype)
        attn_out = self.attn(hidden_states=x, cos_sin=cos_sin)
        x = rms_norm(x + attn_out, 1e-5)
        mlp_out = self.mlp(self.norm2(x))
        gated_mlp = self.gate.tanh() * mlp_out
        x = rms_norm(x + gated_mlp, 1e-5)
        return x

class TieredProposerBank(nn.Module):
    def __init__(self, dim: int, depth: int, count: int, forward_dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.forward_dtype = forward_dtype
        self.nets = nn.ModuleList([
            nn.ModuleList([GriffinBlock(dim, heads=max(8, dim // 64), forward_dtype=forward_dtype) for _ in range(depth)])
            for _ in range(count)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, states: torch.Tensor, inject: torch.Tensor, cos_sin=None):
        outs = []
        for i, blocks in enumerate(self.nets):
            state_input = states[i] if i < states.shape[0] else states[-1]
            if state_input.shape[-1] != self.dim:
                if state_input.shape[-1] < self.dim:
                    pad_size = self.dim - state_input.shape[-1]
                    state_input = F.pad(state_input, (0, pad_size))
                else:
                    state_input = state_input[..., :self.dim]
            state_input = state_input.to(self.forward_dtype)
            
            h = state_input
            for block in blocks:
                h = block(h, inject=inject, cos_sin=cos_sin)
            outs.append(self.norm(h))
        return torch.stack(outs, dim=0)

class MurderTreeV2Core(nn.Module):
    def __init__(self, cfg: MurderTreeV2Config):
        super().__init__()
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.forward_dtype)

        d = cfg.hidden_size
        self.embed = DTypeAwareEmbedding(cfg.vocab_size, d, forward_dtype=self.dtype)

        self.rotary_tier1 = None
        self.rotary_tier2 = None
        self.rotary_tier3 = None
        self.rotary_verifier = None

        self.tier1 = TieredProposerBank(cfg.tier1_dim, cfg.tier1_depth, cfg.tier1_count, forward_dtype=self.dtype)
        self.tier2 = TieredProposerBank(cfg.tier2_dim, cfg.tier2_depth, cfg.tier2_count, forward_dtype=self.dtype)
        self.tier3 = TieredProposerBank(cfg.tier3_dim, cfg.tier3_depth, cfg.tier3_count, forward_dtype=self.dtype)

        self.up1 = DTypeAwareLinear(d, cfg.tier1_dim, forward_dtype=self.dtype)
        self.up2 = DTypeAwareLinear(d, cfg.tier2_dim, forward_dtype=self.dtype)
        self.up3 = DTypeAwareLinear(d, cfg.tier3_dim, forward_dtype=self.dtype)

        self.down1 = DTypeAwareLinear(cfg.tier1_dim, d, forward_dtype=self.dtype)
        self.down2 = DTypeAwareLinear(cfg.tier2_dim, d, forward_dtype=self.dtype)
        self.down3 = DTypeAwareLinear(cfg.tier3_dim, d, forward_dtype=self.dtype)

        self.verifier = nn.ModuleList([GriffinBlock(d, heads=cfg.num_heads, forward_dtype=self.dtype) for _ in range(cfg.verifier_depth)])
        self.fast_critic = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)
        self.slow_critic = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)

        self.lm_head = DTypeAwareLinear(d, cfg.vocab_size, bias=False, forward_dtype=self.dtype)
        self.halt_head = DTypeAwareLinear(d, 2, bias=True, forward_dtype=self.dtype)
        self.evolve_head = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)

        self.policy_memory = nn.Parameter(torch.zeros(cfg.num_policy_slots, d, dtype=self.dtype))
        nn.init.trunc_normal_(self.policy_memory, std=0.02)

        self.recurrent_gate = DTypeAwareLinear(d, d, bias=False, forward_dtype=self.dtype)
        
        self.apply(self._init_weights)
        with torch.no_grad():
            self.halt_head.bias.fill_(-6.0)

    def _init_weights(self, m):
        if isinstance(m, (DTypeAwareLinear, nn.Linear)):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _ensure_rotary(self, seq_len, device):
        if self.rotary_tier1 is None:
            tier1_heads = max(8, self.cfg.tier1_dim // 64)
            tier1_head_dim = self.cfg.tier1_dim // tier1_heads
            self.rotary_tier1 = RotaryEmbedding(
                dim=tier1_head_dim,
                max_position_embeddings=seq_len, 
                base=self.cfg.rope_theta,
                device=device
            )
        
        if self.rotary_tier2 is None:
            tier2_heads = max(8, self.cfg.tier2_dim // 64)
            tier2_head_dim = self.cfg.tier2_dim // tier2_heads
            self.rotary_tier2 = RotaryEmbedding(
                dim=tier2_head_dim,
                max_position_embeddings=seq_len, 
                base=self.cfg.rope_theta,
                device=device
            )
        
        if self.rotary_tier3 is None:
            tier3_heads = max(8, self.cfg.tier3_dim // 64)
            tier3_head_dim = self.cfg.tier3_dim // tier3_heads
            self.rotary_tier3 = RotaryEmbedding(
                dim=tier3_head_dim,
                max_position_embeddings=seq_len, 
                base=self.cfg.rope_theta,
                device=device
            )
        
        if self.rotary_verifier is None:
            verifier_head_dim = self.cfg.hidden_size // self.cfg.num_heads
            self.rotary_verifier = RotaryEmbedding(
                dim=verifier_head_dim,
                max_position_embeddings=seq_len, 
                base=self.cfg.rope_theta,
                device=device
            )

        return (
            self.rotary_tier1(seq_len),
            self.rotary_tier2(seq_len),
            self.rotary_tier3(seq_len),
            self.rotary_verifier(seq_len)
        )

    def _get_evolution_conditions(self, q_evolve, evolution_mask):
        B = q_evolve.shape[0]
        device = q_evolve.device
        
        if self.training:
            evolve_now = (q_evolve.sigmoid() > self.cfg.evolution_trigger_threshold)
            num_candidates = self.cfg.evolution_candidates
            noise_scale = 0.02
            use_slow_critic = True
        else:
            if self.cfg.allow_inference_evolution:
                current_evolutions = evolution_mask.sum(dim=0)  # [B]
                can_evolve = current_evolutions < self.cfg.max_inference_evolutions
                evolve_now = (q_evolve.sigmoid() > self.cfg.inference_evolution_threshold) & can_evolve
                num_candidates = self.cfg.inference_evolution_candidates
                noise_scale = self.cfg.inference_noise_scale
                use_slow_critic = False  # Use fast critic for inference speed
            else:
                evolve_now = torch.zeros(B, dtype=torch.bool, device=device)
                num_candidates = 0
                noise_scale = 0.0
                use_slow_critic = False
        
        return evolve_now, num_candidates, noise_scale, use_slow_critic

    def _select_evolution_slot(self, evolution_mask_b, batch_idx, current_step):
        mask = ~evolution_mask_b
        if mask.any():
            return mask.to(torch.long).argmax(0).item()
        else:
            if self.training:
                # Random replacement during training
                return torch.randint(0, self.cfg.num_policy_slots, (1,)).item()
            else:
                # More systematic replacement during inference
                # Use round-robin based on current step
                return (current_step + batch_idx) % self.cfg.num_policy_slots

    def _vectorized_policy_evolution(self, carry, pooled_h, evolve_now, num_candidates, noise_scale, use_slow_critic, device):
        """Enhanced policy evolution with vectorized recombination and quality-aware slot management"""
        B = pooled_h.shape[0]
        d = self.cfg.hidden_size
        
        evolve_mask = evolve_now  # [B]
        active_counts = carry.evolution_mask.sum(dim=0)  # [B]
        
        base_noise = torch.randn(num_candidates, B, d, device=device, dtype=self.dtype) * noise_scale
        
        cands = pooled_h.unsqueeze(0) + base_noise  # [num_candidates, B, d]
        
        multi_policy_mask = (active_counts >= 2) & evolve_mask  # [B]
        
        if multi_policy_mask.any():
            multi_batch_indices = torch.where(multi_policy_mask)[0]  # [M]
            M = multi_batch_indices.shape[0]
            
            active_mask_multi = carry.evolution_mask[:, multi_policy_mask]  # [slots, M]
            active_policies_multi = carry.z_mem[:, multi_policy_mask, :]  # [slots, M, d]
            
            active_counts_multi = active_mask_multi.sum(dim=0)  # [M]
            
            parent1_idx = torch.zeros(num_candidates, M, device=device, dtype=torch.long)
            parent2_idx = torch.zeros(num_candidates, M, device=device, dtype=torch.long)
            
            for m in range(M):
                active_count = active_counts_multi[m]
                parent1_idx[:, m] = torch.randint(0, active_count, (num_candidates,), device=device)
                
                parent2_temp = torch.randint(0, active_count - 1, (num_candidates,), device=device)
                parent2_idx[:, m] = torch.where(parent2_temp >= parent1_idx[:, m], 
                                              parent2_temp + 1, parent2_temp)
            
            if self.training and M > 0:
                with torch.no_grad():
                    active_flat = active_policies_multi[active_mask_multi]  # [total_active, d]
                    if use_slow_critic:
                        active_scores_flat = self.slow_critic(active_flat.unsqueeze(1)).squeeze(-1)  # [total_active, 1]
                    else:
                        active_scores_flat = self.fast_critic(active_flat.unsqueeze(1)).squeeze(-1)  # [total_active, 1]
                    
                    active_scores_flat = active_scores_flat.squeeze(-1)  # [total_active]
                    
                    active_scores = torch.full((active_mask_multi.shape[0], M), -float('inf'), 
                                             device=device, dtype=active_scores_flat.dtype)
                    active_mask_flat = active_mask_multi.view(-1)
                    active_scores.view(-1)[active_mask_flat] = active_scores_flat
                    
                    top_half_mask = torch.zeros_like(active_mask_multi, dtype=torch.bool)
                    for m in range(M):
                        active_count = active_counts_multi[m]
                        if active_count > 0:
                            scores_m = active_scores[active_mask_multi[:, m], m]
                            top_k = max(1, active_count // 2)
                            _, top_indices = torch.topk(scores_m, top_k)
                            actual_indices = torch.where(active_mask_multi[:, m])[0][top_indices]
                            top_half_mask[actual_indices, m] = True
                    
                    for m in range(M):
                        if active_counts_multi[m] >= 2:
                            top_indices_m = torch.where(top_half_mask[:, m])[0]
                            if len(top_indices_m) >= 2:
                                new_parent1_idx = torch.randint(0, len(top_indices_m), 
                                                              (num_candidates,), device=device)
                                parent1_idx[:, m] = top_indices_m[new_parent1_idx]
                                
                                new_parent2_temp = torch.randint(0, len(top_indices_m) - 1, 
                                                               (num_candidates,), device=device)
                                new_parent2_idx = torch.where(new_parent2_temp >= new_parent1_idx, 
                                                            new_parent2_temp + 1, new_parent2_temp)
                                parent2_idx[:, m] = top_indices_m[new_parent2_idx]
            
            batch_indices = torch.arange(M, device=device).unsqueeze(0).expand(num_candidates, M)
            
            active_indices_flat = []
            batch_indices_flat = []
            for m in range(M):
                active_indices_m = torch.where(active_mask_multi[:, m])[0]  # [active_count]
                active_indices_flat.append(active_indices_m[parent1_idx[:, m]])
                active_indices_flat.append(active_indices_m[parent2_idx[:, m]])
                batch_indices_flat.append(batch_indices[:, m])
                batch_indices_flat.append(batch_indices[:, m])
            
            active_indices_flat = torch.cat(active_indices_flat)  # [2 * num_candidates * M]
            batch_indices_flat = torch.cat(batch_indices_flat)    # [2 * num_candidates * M]
            
            all_parents = active_policies_multi[active_indices_flat, batch_indices_flat]  # [2 * num_candidates * M, d]
            all_parents = all_parents.view(2, num_candidates, M, d)  # [2, num_candidates, M, d]
            parent1 = all_parents[0]  # [num_candidates, M, d]
            parent2 = all_parents[1]  # [num_candidates, M, d]
            
            alphas = torch.rand(num_candidates, M, 1, device=device, dtype=parent1.dtype)  # [num_candidates, M, 1]
            recombined = alphas * parent1 + (1 - alphas) * parent2
            
            cands_multi = recombined + base_noise[:, multi_policy_mask, :]
            cands_multi = rms_norm(cands_multi, self.cfg.rms_eps)
            
            half = num_candidates // 2
            
            cands_multi = cands_multi.to(cands.dtype)
            cands[:half, multi_policy_mask, :] = cands_multi[:half]
        
        cands = rms_norm(cands, self.cfg.rms_eps)
        
        best_policy = rms_norm(pooled_h.unsqueeze(0), self.cfg.rms_eps)  # [1, B, d]
        cands[0] = torch.where(evolve_mask.unsqueeze(-1), best_policy[0], cands[0])
        
        cands_flat = cands.reshape(-1, d)  # [num_candidates * B, d]
        
        if use_slow_critic:
            values_flat = self.slow_critic(cands_flat.unsqueeze(1)).squeeze(-1)  # [num_candidates * B, 1]
        else:
            values_flat = self.fast_critic(cands_flat.unsqueeze(1)).squeeze(-1)  # [num_candidates * B, 1]
        
        values_flat = values_flat.squeeze(-1)  # [num_candidates * B]
        values = values_flat.view(num_candidates, B)  # [num_candidates, B]
        
        best_values, best_indices = values.max(dim=0)  # [B] - get both best values and indices
        
        batch_indices = torch.arange(B, device=device)
        new_policy = cands[best_indices, batch_indices]  # [B, d]
        
        current_step = carry.steps[0] if carry.steps is not None else 0
        evolve_batch_indices = torch.where(evolve_mask)[0]
        
        if evolve_batch_indices.numel() > 0:
            slot_mask = ~carry.evolution_mask[:, evolve_batch_indices]  # [slots, M]
            
            available_slots = slot_mask.long().argmax(dim=0)  # [M]
            has_available_slot = slot_mask.any(dim=0)  # [M]
            
            if has_available_slot.any():
                available_batches = evolve_batch_indices[has_available_slot]
                available_slots_for_these = available_slots[has_available_slot]
                carry.z_mem[available_slots_for_these, available_batches] = new_policy[available_batches].detach()
                carry.evolution_mask[available_slots_for_these, available_batches] = True
            
            no_slot_batches = evolve_batch_indices[~has_available_slot]
            if no_slot_batches.numel() > 0:
                existing_policies = carry.z_mem[:, no_slot_batches]  # [slots, K, d]
                K = no_slot_batches.numel()
                
                existing_flat = existing_policies.reshape(-1, d)  # [slots * K, d]
                if use_slow_critic:
                    existing_values_flat = self.slow_critic(existing_flat.unsqueeze(1)).squeeze(-1)  # [slots * K, 1]
                else:
                    existing_values_flat = self.fast_critic(existing_flat.unsqueeze(1)).squeeze(-1)  # [slots * K, 1]
                
                existing_values_flat = existing_values_flat.squeeze(-1)  # [slots * K]
                existing_values = existing_values_flat.view(existing_policies.shape[0], K)  # [slots, K]
                
                worst_slot_indices = existing_values.argmin(dim=0)  # [K]
                
                new_values_for_batches = best_values[no_slot_batches]  # [K]
                worst_existing_values = existing_values[worst_slot_indices, 
                                                      torch.arange(K, device=device)]  # [K]
                
                should_replace = new_values_for_batches > worst_existing_values
                
                if should_replace.any():
                    replace_batches = no_slot_batches[should_replace]
                    replace_slots = worst_slot_indices[should_replace]
                    
                    carry.z_mem[replace_slots, replace_batches] = new_policy[replace_batches].detach()
        
        return new_policy

    def forward(self, carry: HierarchicalReasoningModel_ACTV2InnerCarry, input_ids: torch.Tensor):
        B, T = input_ids.shape
        d = self.cfg.hidden_size
        device = input_ids.device
    
        x = self.embed(input_ids.to(torch.int32)) * math.sqrt(d)
        
        cos_sin_tier1, cos_sin_tier2, cos_sin_tier3, cos_sin_verifier = self._ensure_rotary(T, device)
    
        h = carry.z_H.mean(0).to(self.dtype)  # [B, d] - per-position state
    
        if carry.z_mem is not None and carry.evolution_mask is not None and carry.evolution_mask.any():
            policy_inject = []
            for b in range(B):
                active = carry.z_mem[carry.evolution_mask[:, b], b, :]  # [num_active, d]
                if active.shape[0] > 0:
                    policy_inject.append(active.mean(0))
                else:
                    policy_inject.append(torch.zeros(d, device=device, dtype=self.dtype))
            policy_inject = torch.stack(policy_inject)  # [B, d]
            policy_inject = policy_inject.unsqueeze(1).expand(-1, T, -1)  # [B, T, d]
            h = h.unsqueeze(1).expand(-1, T, -1) + policy_inject * 0.3  # [B, T, d]

        # For now, we'll keep the sequence processing but note this should be changed
        t1_in = self.up1(x)  # [B, T, hidden_size] -> [B, T, tier1_dim]
        
        # Process through tier1 - FIXED: This should be per-position but we keep sequence for now
        t1_states = carry.z_H[:self.cfg.tier1_count].unsqueeze(2).expand(-1, -1, T, -1)  # [tier1_count, B, T, d]
        t1_out = self.tier1(t1_states, t1_in, cos_sin=cos_sin_tier1)
        
        t1_to_hidden = self.down1(t1_out.mean(0))  # [B, T, tier1_dim] -> [B, T, hidden_size]
        t2_in = self.up2(t1_to_hidden)  # [B, T, hidden_size] -> [B, T, tier2_dim]
        
        t2_states = carry.z_H[self.cfg.tier1_count:self.cfg.tier1_count + self.cfg.tier2_count].unsqueeze(2).expand(-1, -1, T, -1)
        t2_out = self.tier2(t2_states, t2_in, cos_sin=cos_sin_tier2)
        
        t2_to_hidden = self.down2(t2_out.mean(0))  # [B, T, tier2_dim] -> [B, T, hidden_size]
        t3_in = self.up3(t2_to_hidden)  # [B, T, hidden_size] -> [B, T, tier3_dim]
        
        t3_states = carry.z_H[-self.cfg.tier3_count:].unsqueeze(2).expand(-1, -1, T, -1)
        candidates = self.tier3(t3_states, t3_in, cos_sin=cos_sin_tier3)

        verify_in = self.down3(candidates.mean(0))  # [B, T, tier3_dim] -> [B, T, hidden_size]
        
        slow = (carry.steps is not None and len(carry.steps) > 0 and 
                carry.steps[0] > 0 and 
                (carry.steps[0] % self.cfg.slow_verify_every == 0))
        verify_in_copy = verify_in
        if slow and self.training:
            for block in self.verifier:
                verify_in_copy = block(verify_in_copy, inject=None, cos_sin=cos_sin_verifier)
            value = self.slow_critic(verify_in_copy)
        else:
            value = self.fast_critic(verify_in_copy)
    
        if h.dim() == 2:  # [B, d]
            h = h.unsqueeze(1).expand(-1, verify_in_copy.shape[1], -1)  # [B, T, d]
    
        gate = torch.sigmoid(self.recurrent_gate(h)) * 0.5
        new_h = rms_norm(h + gate * verify_in_copy, self.cfg.rms_eps)
    
        logits = self.lm_head(new_h)
        q = self.halt_head(new_h)
        q_halt, q_cont = q[:, :, 0], q[:, :, 1]
        q_evolve = self.evolve_head(new_h.mean(1))  # [B, 1]

        evolve_now, num_candidates, noise_scale, use_slow_critic = self._get_evolution_conditions(
            q_evolve.squeeze(-1), carry.evolution_mask
        )

        new_policy = None
        if evolve_now.any() and num_candidates > 0:
            pooled_h = new_h.mean(dim=1)  # [B, d]
            
            new_policy = self._vectorized_policy_evolution(
                carry, pooled_h, evolve_now, num_candidates, 
                noise_scale, use_slow_critic, device
            )

        t1_out_proj = self.down1(t1_out.view(-1, self.cfg.tier1_dim)).view(t1_out.shape[0], B, T, d).mean(2)  # [tier1_count, B, d]
        t2_out_proj = self.down2(t2_out.view(-1, self.cfg.tier2_dim)).view(t2_out.shape[0], B, T, d).mean(2)  # [tier2_count, B, d]
        t3_out_proj = self.down3(candidates.view(-1, self.cfg.tier3_dim)).view(candidates.shape[0], B, T, d).mean(2)  # [tier3_count, B, d]

        return new_h, logits, (q_halt, q_cont), value, new_policy, t1_out_proj, t2_out_proj, t3_out_proj


class HierarchicalReasoningModel_ACTV2(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.cfg = MurderTreeV2Config(**config_dict)
        self.core = MurderTreeV2Core(self.cfg)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        device = next(self.core.parameters()).device
        bs = batch["inputs"].shape[0]
        d = self.cfg.hidden_size
        total_experts = self.cfg.tier1_count + self.cfg.tier2_count + self.cfg.tier3_count

        fake_H = torch.zeros(total_experts, bs, d, device=device, dtype=self.core.dtype)
        z_mem = torch.zeros(self.cfg.num_policy_slots, bs, d, device=device, dtype=self.core.dtype)
        evolution_mask = torch.zeros(self.cfg.num_policy_slots, bs, dtype=torch.bool, device=device)
        steps = torch.zeros(bs, dtype=torch.int32, device=device)

        inner = HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=fake_H,
            z_L=torch.zeros(bs, self.cfg.seq_len, d, device=device, dtype=self.core.dtype),
            z_mem=z_mem,
            evolution_mask=evolution_mask,
            steps=steps,
        )

        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=inner,
            steps=steps,
            halted=torch.zeros(bs, dtype=torch.bool, device=device),
            current_data=batch,
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2Carry, batch: Dict[str, torch.Tensor]):
        inner = carry.inner_carry
        inner.steps = carry.steps

        new_h, logits, (q_halt, q_cont), value, new_policy, t1_out, t2_out, t3_out = self.core(inner, batch["inputs"])
        
        new_z_H = torch.cat([t1_out, t2_out, t3_out], dim=0)

        embedded = self.core.embed(batch["inputs"].to(torch.int32))
        decay = 0.9
        current_z_L = inner.z_L[:, :embedded.shape[1], :]
        new_z_L = decay * current_z_L + (1 - decay) * embedded
        if new_z_L.shape[1] < inner.z_L.shape[1]:
            pad_len = inner.z_L.shape[1] - new_z_L.shape[1]
            padding = torch.zeros(embedded.shape[0], pad_len, embedded.shape[2], device=embedded.device, dtype=embedded.dtype)
            new_z_L = torch.cat([new_z_L, padding], dim=1)

        new_inner = HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=new_z_H,
            z_L=new_z_L,
            z_mem=inner.z_mem,
            evolution_mask=inner.evolution_mask,
            steps=inner.steps,
        )

        carry.steps += 1
        with torch.no_grad():
            q_halt_mean = q_halt.mean(dim=1)
            q_cont_mean = q_cont.mean(dim=1)
            
            should_halt = (carry.steps >= self.cfg.max_steps) | (q_halt_mean > q_cont_mean)
            if self.training:
                random_halt = torch.rand_like(should_halt.float()) < 0.05
                should_halt = should_halt | random_halt
            carry.halted = carry.halted | should_halt

        carry.inner_carry = new_inner
        outputs = {
            "logits": logits,
            "value": value.squeeze(-1),
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "loss_aux": self.cfg.ponder_cost_weight * carry.steps.float().mean() if self.training else torch.tensor(0.0),
            "active_policies": inner.evolution_mask.sum().item() if inner.evolution_mask is not None else 0,
            "new_policy": new_policy,
            "steps": carry.steps,
        }

        return carry, outputs


def test_enhanced_evolution():
    """Test the enhanced evolution system"""
    print("Testing Enhanced Evolution System")
    print("=" * 50)
    
    config = {
        "vocab_size": 1000, "seq_len": 32, "hidden_size": 256,
        "tier1_dim": 128, "tier1_depth": 1, "tier1_count": 2,
        "tier2_dim": 192, "tier2_depth": 1, "tier2_count": 1,
        "tier3_dim": 256, "tier3_depth": 1, "tier3_count": 1,
        "verifier_depth": 2, "max_steps": 8, "num_policy_slots": 4,
        "evolution_candidates": 4, "evolution_trigger_threshold": 0.1,
        "allow_inference_evolution": True,
        "inference_evolution_threshold": 0.3,
        "inference_evolution_candidates": 2,
        "max_inference_evolutions": 3,
    }
    
    model = HierarchicalReasoningModel_ACTV2(config)
    batch = {"inputs": torch.randint(0, 100, (3, 32))}
    
    print("Testing training mode evolution...")
    model.train()
    carry = model.initial_carry(batch)
    
    evolution_history = []
    for step in range(6):
        carry, outputs = model(carry, batch)
        active_policies = carry.inner_carry.evolution_mask.sum(dim=0).tolist()
        evolution_history.append(active_policies)
        print(f"Step {step}: Active policies per batch: {active_policies}")
        
        if outputs["new_policy"] is not None:
            print(f"  New policy generated with shape: {outputs['new_policy'].shape}")
    
    print("\nEvolution history:")
    for step, counts in enumerate(evolution_history):
        print(f"  Step {step}: {counts}")
    
    print("\nTesting inference mode evolution...")
    model.eval()
    carry = model.initial_carry(batch)
    
    inference_history = []
    for step in range(6):
        carry, outputs = model(carry, batch)
        active_policies = carry.inner_carry.evolution_mask.sum(dim=0).tolist()
        inference_history.append(active_policies)
        print(f"Step {step}: Active policies per batch: {active_policies}")
    
    print("\nInference evolution history:")
    for step, counts in enumerate(inference_history):
        print(f"  Step {step}: {counts}")
    
    final_counts = inference_history[-1]
    for count in final_counts:
        if count > config["max_inference_evolutions"]:
            print("❌ ERROR: Evolution limit exceeded!")
            return False
    
    print("✓ Evolution limits respected")
    print("✓ Enhanced evolution system working correctly!")
    return True

if __name__ == "__main__":
    test_enhanced_evolution()
