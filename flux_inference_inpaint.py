
import argparse
import datetime
import math
import os
import random
from typing import Callable, List, Optional
import einops
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import accelerate
from transformers import CLIPTextModel
from safetensors.torch import load_file
import json

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux, clora_flux
import networks.clora_engine as clora_engine

init_ipex()

from library import flux_models, flux_utils, strategy_flux
from library.utils import str_to_dtype
import logging

logger = logging.getLogger(__name__)

def get_schedule(num_steps: int, image_seq_len: int, shift: bool = True) -> list[float]:
    # Simple schedule generator mimicking the one in minimal inference
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        # standard shift for Flux
        base_shift = 0.5
        max_shift = 1.15
        x1, y1, x2, y2 = 256, base_shift, 4096, max_shift
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        mu = m * image_seq_len + b
        timesteps = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** 1.0)
    return timesteps.tolist()

def denoise_inpaint(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
    cancel_flag: Optional[list] = [],
    inpaint_latents: Optional[torch.Tensor] = None,
    inpaint_masks: Optional[torch.Tensor] = None, # Combined packed mask [b, seq_len, 1]
    initial_noise: Optional[torch.Tensor] = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    for step_idx, (t_curr, t_prev) in enumerate(zip(tqdm(timesteps[:-1]), timesteps[1:])):
        if cancel_flag and cancel_flag[0]:
            logger.info("Operation cancelled")
            return None
        
        # 1. Apply Inpainting (Force Mixed State) BEFORE prediction step?
        # Or usually standard re-noising is done: z_known = x_0 * (1-t) + noise * t
        # In flux (t goes 1->0): z_t = t * noise + (1-t) * image
        # So we want to replace the current 'img' (which is z_t estimate) with the ground truth z_t in unmasked areas.
        
        if inpaint_latents is not None and inpaint_masks is not None and initial_noise is not None:
            # Calculate ground truth noisy latent at current timestep t_curr
            # t_curr is scalar float.
            t = t_curr
            noisy_ground_truth = t * initial_noise + (1.0 - t) * inpaint_latents
            
            # Blend: img = mask * img + (1-mask) * noisy_ground_truth
            # Mask is 1 for INPAINT (keep img), 0 for KEEP ORIGINAL (use noisy_ground_truth)
            img = inpaint_masks * img + (1.0 - inpaint_masks) * noisy_ground_truth

        # 2. Predict
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        with torch.no_grad():
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                txt_attn_mask=t5_attn_mask,
            )

        # 3. Step (Euler)
        img = img + (t_prev - t_curr) * pred
        
    return img

def pack_latents_flux(latents):
    # latents: [b, c, h, w]
    # packed: [b, (h/2 * w/2), c*4]
    return einops.rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

def unpack_latents_flux(packed, h, w):
    # packed: [b, seq, c_packed] -> [b, c, h, w]
    return einops.rearrange(
        packed, 
        "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
        h=h//2, 
        w=w//2, 
        ph=2, 
        pw=2
    )

class FluxInpaintPipeline:
    def __init__(self, ckpt_path, clip_l_path, t5xxl_path, ae_path, device="cuda", dtype="bfloat16"):
        self.device = torch.device(device)
        self.dtype = str_to_dtype(dtype)
        self.flux_dtype = self.dtype # Simplified
        
        # Load models
        logger.info(f"Loading Flux models...")
        # clip_l can be None for Flux 2 / Mistral mode
        if clip_l_path:
            self.clip_l = flux_utils.load_clip_l(clip_l_path, self.dtype, self.device)
        else:
            self.clip_l = None
        
        # Detect Mistral / Flux 2 mode BEFORE loading text encoder
        self.use_mistral = False
        self.t5xxl = None
        self.mistral_model = None
        self.tokenizer = None
        
        if t5xxl_path and ("mistral" in t5xxl_path.lower() or (clip_l_path is None and t5xxl_path.endswith(".safetensors"))):
            logger.info("Detected Mistral model in t5xxl_path, enabling Flux 2 Mistral mode.")
            self.use_mistral = True
            from transformers import AutoModel, AutoTokenizer
            
            # For Mistral: use the standard Mistral tokenizer since safetensors file doesn't include it
            try:
                # Try loading from directory first
                tokenizer_dir = os.path.dirname(t5xxl_path)
                if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")) or os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
                else:
                    # Fallback to standard Mistral tokenizer
                    logger.info("No local tokenizer found, using mistralai/Mistral-7B-v0.1 tokenizer")
                    self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, falling back to standard Mistral")
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load Mistral model using custom loader
            logger.info("Loading Mistral model for FLUX 2...")
            try:
                self.mistral_model = flux_utils.load_mistral_model(t5xxl_path, self.dtype, self.device)
                # Offload Mistral to CPU to save VRAM for Flux loading
                self.mistral_model.to("cpu")
                self.t5xxl = None
            except Exception as e:
                logger.error(f"Failed to load Mistral encoder: {e}")
                raise
        else:
            # Standard FLUX 1 mode - load T5
            self.t5xxl = flux_utils.load_t5xxl(t5xxl_path, self.dtype, self.device)
            # Offload T5 to CPU
            if self.t5xxl:
                self.t5xxl.to("cpu")
        
        # Clean memory
        torch.cuda.empty_cache()

        self.ae = flux_utils.load_ae(ae_path, self.dtype, self.device, is_flux2=self.use_mistral)
        # Offload AE to CPU
        self.ae.to("cpu")
        
        # Clean memory again before big model load
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load Flux model to CPU initially to save VRAM
        is_schnell, self.model = flux_utils.load_flow_model(ckpt_path, self.flux_dtype, "cpu")
        # Model weights are on CPU
        self.model.eval()
        self.is_schnell = is_schnell
        
        # Strategies
        t5xxl_max_length = 256 if is_schnell else 512
        self.tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
        self.encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

        
    def generate(
        self,
        input_image: Image.Image,
        mask_image: Image.Image, # Single combined mask for now
        prompt: str,
        steps: int = 28,
        guidance: float = 3.5,
        seed: int = None,
        cancel_flag: list = []
    ):
        width, height = input_image.size
        # Resize to pad 16
        w = max(64, (width // 16) * 16)
        h = max(64, (height // 16) * 16)
        if w!=width or h!=height:
             input_image = input_image.resize((w, h), Image.LANCZOS)
             mask_image = mask_image.resize((w, h), Image.NEAREST)
        
        # Encode Image
        img_tensor = torch.from_numpy(np.array(input_image)).float() / 127.5 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.ae.dtype)
        
        # Offload: Move AE to GPU
        self.ae.to(self.device)
        with torch.no_grad():
             latents = self.ae.encode(img_tensor)
        # Offload: Move AE back to CPU
        self.ae.to("cpu")
        torch.cuda.empty_cache()
        packed_latents = pack_latents_flux(latents).to(self.flux_dtype)
        
        # Process Mask
        m_tensor = torch.from_numpy(np.array(mask_image.convert("L"))).float() / 255.0
        m_tensor = m_tensor.unsqueeze(0).unsqueeze(0).to(self.device).to(self.flux_dtype) # [1,1,h,w]
        
        # Downsample mask to latent size (h/8, w/8)? 
        # Wait, Flux AE downsample factor?
        # Standard SD is 8. But packing function implies 16?
        # If I look at minimal inference: 
        # packed_latent_height = math.ceil(image_height / 16)
        # So effective stride is 16.
        # But AE encode output size? 
        # Let's trust `latents.shape`.
        
        latent_h, latent_w = latents.shape[2], latents.shape[3]
        m_tensor = torch.nn.functional.interpolate(m_tensor, size=(latent_h, latent_w), mode="nearest")
        
        # Pack mask
        # Expand channels to match latent channels? 
        # Latents [b, 16, lh, lw] ?
        # Packed [b, (lh/2*lw/2), 64]
        # We need mask [b, (lh/2*lw/2), 1] or broadcastable.
        
        # Check latents shape
        # Assuming VAE output is 4 channels for standard SD, 16 for Flux?
        # If Flux VAE is 16 channels, then packed is 64 channels.
        # If we interpolate mask to latent_h, latent_w, then rearrange like latents:
        # m_packed = rearrange(m_tensor, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        # m_packed shape will be [1, (token_count), 4] (since 1 channel * 2 * 2 = 4)
        # We want to know if a patch is masked or not.
        # Let's take max to be safe (if any part is masked, treat as masked)
        m_packed = einops.rearrange(m_tensor, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        m_packed = torch.max(m_packed, dim=2, keepdim=True)[0] # [1, tokens, 1]
        m_packed = (m_packed > 0.5).to(self.flux_dtype)
        
        # Create Noise
        if seed is None: seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(packed_latents.shape, device=self.device, dtype=self.flux_dtype, generator=generator)
        
        # Prepare Prompt
        if self.use_mistral:
             # Custom Mistral Encoding
             # Offload: Move Mistral to GPU
             self.mistral_model.to(self.device)
             with torch.no_grad():
                 inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)
                 # Request hidden states to allow concatenation if needed
                 outputs = self.mistral_model(**inputs, output_hidden_states=True)
                 
                 # hidden_states: [B, Seq, Dim]
                 # for Flux 2 with context_in_dim 15360, we likely need to concat last 3 layers (5120*3)
                 txt = outputs.last_hidden_state.to(self.flux_dtype)
                 
                 if hasattr(self.model, 'params') and self.model.params.context_in_dim == 15360 and txt.shape[-1] == 5120:
                     # Concatenate last 3 hidden states
                     h_states = outputs.hidden_states
                     # Check if we have enough states
                     if len(h_states) >= 3:
                         txt = torch.cat([h_states[-3], h_states[-2], h_states[-1]], dim=-1).to(self.flux_dtype)
                         print(f"Concatenated 3 Mistral hidden states to shape {txt.shape}")
                     else:
                         print(f"Warning: Not enough hidden states ({len(h_states)}) to concat for Flux 2")
                         # Fallback: repeat?
                         txt = torch.cat([txt, txt, txt], dim=-1)
                 elif hasattr(self.model, 'params') and self.model.params.context_in_dim != txt.shape[-1]:
                     print(f"Warning: Mistral output dim {txt.shape[-1]} != Model context dim {self.model.params.context_in_dim}")
             
             # Offload: Move Mistral back to CPU
             self.mistral_model.to("cpu")
             torch.cuda.empty_cache()
                 
             # Pool for 'vec'?
             # Flux 2 might use EOS token or Mean.
             # Let's check UNet expected dim.
             if hasattr(self.model, 'params'):
                  expected_vec_dim = self.model.params.vec_in_dim
             else:
                  expected_vec_dim = 768 # Default Flux 1
             
             if expected_vec_dim > 1000: # Context dim usually > 1000
                  # If vec_in is same as context, use pooled or mean
                  vec = txt.mean(dim=1)
             else:
                  # If mismatch (e.g. 4096 vs 768), we can't easily project without weights.
                  # Assuming Flux 2 UNet matches Mistral dim (4096?) or allows it.
                  # Or maybe vec is 0?
                  vec = torch.zeros((txt.shape[0], expected_vec_dim), device=self.device, dtype=self.flux_dtype)
             
             t5_out = txt
             txt_ids = torch.zeros(txt.shape[0], txt.shape[1], 3, device=self.device, dtype=self.flux_dtype)
             t5_attn_mask = inputs.attention_mask.bool()
             l_pooled = vec

        else:
             # Standard Flux
             tokens_and_masks = self.tokenize_strategy.tokenize(prompt)
             with torch.no_grad():
                  l_pooled, t5_out, txt_ids, t5_attn_mask = self.encoding_strategy.encode_tokens(
                       self.tokenize_strategy, 
                       [self.clip_l, self.t5xxl], 
                       tokens_and_masks
                  )
        
        # Prepare Img IDs
        # simplified, use utility
        img_ids = flux_utils.prepare_img_ids(1, latent_h, latent_w).to(self.device).to(self.flux_dtype)
        
        # Denoise
        timesteps = get_schedule(steps, packed_latents.shape[1], shift=not self.is_schnell)
        
        # Offload: Move Flux to GPU
        print("Moving Flux model to GPU...")
        self.model.to(self.device)
        
        try:
            x = denoise_inpaint(
                 self.model,
                 noise,
                 img_ids,
                 t5_out,
                 txt_ids,
                 l_pooled,
                 timesteps,
                 guidance=guidance,
                 t5_attn_mask=t5_attn_mask,
                 cancel_flag=cancel_flag,
                 inpaint_latents=packed_latents,
                 inpaint_masks=m_packed,
                 initial_noise=noise
            )
        finally:
            # Offload: Move Flux back to CPU
            print("Moving Flux model back to CPU...")
            self.model.to("cpu")
            torch.cuda.empty_cache()
        
        if x is None: return None
        
        # Decode
        # unpack expects latent height/width, the function will multiply by 2 internally due to ph=2, pw=2
        x = unpack_latents_flux(x, latent_h, latent_w)
        # Offload: Move AE to GPU
        self.ae.to(self.device)
        with torch.no_grad():
             x = self.ae.decode(x)
        # Offload: Move AE back to CPU
        self.ae.to("cpu")
        torch.cuda.empty_cache()
             
        x = x.clamp(-1, 1)
        x = x.permute(0, 2, 3, 1)
        img_out = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])
        return img_out

if __name__ == "__main__":
    # Simple CLI test
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_l", type=str, required=True)
    parser.add_argument("--t5xxl", type=str, required=True)
    parser.add_argument("--ae", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    
    pipeline = FluxInpaintPipeline(args.ckpt_path, args.clip_l, args.t5xxl, args.ae)
    img = Image.open(args.input)
    mask = Image.open(args.mask)
    res = pipeline.generate(img, mask, args.prompt)
    res.save("inpaint_result.png")
