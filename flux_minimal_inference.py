# Minimum Inference Code for FLUX

import argparse
import datetime
import math
import os
import uuid
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

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
from networks import oft_flux, clora_flux
import networks.clora_engine as clora_engine

init_ipex()


from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    t5_attn_mask: Optional[torch.Tensor] = None,
    neg_txt: Optional[torch.Tensor] = None,
    neg_vec: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
    cancel_flag: Optional[list] = [],
    lora_models: Optional[list] = [],
    clora_mode: str = "soft",
    clora_tau: float = 0.07,
    mask_interval: int = 1,
):
    use_clora = len(lora_models) >= 2
    if use_clora:
        collector = clora_engine.AttentionCollector(model.hidden_size, model.num_heads)
        collector.attach(model)
        # Disable all LoRAs initially; we will enable them one-by-one during inference
        for lm in lora_models:
            if hasattr(lm, "set_enabled"):
                lm.set_enabled(False)
    else:
        collector = None

    try:
        logger.info(f"guidance: {guidance}, cfg_scale: {cfg_scale}")
        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

        if neg_txt is not None and neg_vec is not None:
            b_img_ids = torch.cat([img_ids, img_ids], dim=0)
            b_txt_ids = torch.cat([txt_ids, txt_ids], dim=0)
            b_txt = torch.cat([neg_txt, txt], dim=0)
            b_vec = torch.cat([neg_vec, vec], dim=0)
            b_t5_attn_mask = torch.cat([neg_t5_attn_mask, t5_attn_mask], dim=0) if t5_attn_mask is not None and neg_t5_attn_mask is not None else None
        else:
            b_img_ids, b_txt_ids, b_txt, b_vec, b_t5_attn_mask = img_ids, txt_ids, txt, vec, t5_attn_mask

        current_masks = None
        for step_idx, (t_curr, t_prev) in enumerate(zip(tqdm(timesteps[:-1]), timesteps[1:])):
            if cancel_flag and cancel_flag[0]:
                logger.info("Operation cancelled")
                return None

            t_vec = torch.full((b_img_ids.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            b_img = torch.cat([img, img], dim=0) if neg_txt is not None and neg_vec is not None else img

            if use_clora:
                preds_cpu = {}
                collector.clear()
                for idx, lm in enumerate(lora_models):
                    name = getattr(lm, "lora_name", f"lora_{idx}")
                    collector.set_active(name)
                    lm.set_enabled(True)

                    with torch.no_grad():
                        pred = model(
                            img=b_img,
                            img_ids=b_img_ids,
                            txt=b_txt,
                            txt_ids=b_txt_ids,
                            y=b_vec,
                            timesteps=t_vec,
                            guidance=guidance_vec,
                            txt_attention_mask=b_t5_attn_mask,
                        )

                    lm.set_enabled(False)

                    if neg_txt is not None and neg_vec is not None:
                        pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

                    preds_cpu[name] = pred.cpu()
                    torch.cuda.empty_cache()

                if step_idx % mask_interval == 0 or current_masks is None:
                    current_masks = clora_engine.build_masks(collector.block_sums, mode=clora_mode, tau=clora_tau)
                pred = clora_engine.fuse(preds_cpu, current_masks).to(img.device)

                # Clean up CPU tensors to free memory
                del preds_cpu
            else:
                with torch.no_grad():
                    pred = model(
                        img=b_img,
                        img_ids=b_img_ids,
                        txt=b_txt,
                        txt_ids=b_txt_ids,
                        y=b_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                        txt_attention_mask=b_t5_attn_mask,
                    )

                if neg_txt is not None and neg_vec is not None:
                    pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            img = img + (t_prev - t_curr) * pred

    finally:
        if use_clora and collector is not None:
            collector.detach()

    return img



def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    t5_attn_mask: Optional[torch.Tensor],
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
    neg_l_pooled: Optional[torch.Tensor] = None,
    neg_t5_out: Optional[torch.Tensor] = None,
    neg_t5_attn_mask: Optional[torch.Tensor] = None,
    cfg_scale: Optional[float] = None,
    cancel_flag: Optional[list] = [],
    lora_models: Optional[list] = [],
    clora_mode: str = "soft",
    clora_tau: float = 0.07,
    mask_interval: int = 1,
):
    logger.info(f"num_steps: {num_steps}")
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
                cancel_flag,
                lora_models,
                clora_mode,
                clora_tau,
                mask_interval,
            )
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(
                model,
                img,
                img_ids,
                t5_out,
                txt_ids,
                l_pooled,
                timesteps,
                guidance,
                t5_attn_mask,
                neg_t5_out,
                neg_l_pooled,
                neg_t5_attn_mask,
                cfg_scale,
                cancel_flag,
                lora_models,
                clora_mode,
                clora_tau,
                mask_interval,
            )

    return x


def generate_image(
    model,
    clip_l: CLIPTextModel,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
    negative_prompt: Optional[str],
    cfg_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    clip_l_dtype: torch.dtype,
    t5xxl_dtype: torch.dtype,
    is_schnell: bool,
    flux_dtype: torch.dtype,
    ae_dtype: torch.dtype,
    tokenize_strategy: strategy_flux.FluxTokenizeStrategy,
    encoding_strategy: strategy_flux.TextEncodingStrategy,
    accelerator: accelerate.Accelerator,
    args: Optional[argparse.Namespace] = None,
    cancel_flag: list = [],
    lora_models: Optional[list] = [],
    clora_mode: str = "soft",
    clora_tau: float = 0.07,
    mask_interval: int = 1,
):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(
        image_width / 16
    )
    noise_dtype = torch.float32 if is_fp8(dtype) else dtype
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=noise_dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    # prepare img and img ids

    # this is needed only for img2img
    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #     img = repeat(img, "1 ... -> bs ...", bs=bs)

    # txt2img only needs img_ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

    # prepare fp8 models
    if is_fp8(clip_l_dtype) and (
        not hasattr(clip_l, "fp8_prepared") or not clip_l.fp8_prepared
    ):
        logger.info(
            f"prepare CLIP-L for fp8: set to {clip_l_dtype}, set embeddings to {torch.bfloat16}"
        )
        clip_l.to(clip_l_dtype)  # fp8
        clip_l.text_model.embeddings.to(dtype=torch.bfloat16)
        clip_l.fp8_prepared = True

    if is_fp8(t5xxl_dtype) and (
        not hasattr(t5xxl, "fp8_prepared") or not t5xxl.fp8_prepared
    ):
        logger.info(f"prepare T5xxl for fp8: set to {t5xxl_dtype}")

        def prepare_fp8(text_encoder, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    hidden_gelu = module.act(module.wi_0(hidden_states))
                    hidden_linear = module.wi_1(hidden_states)
                    hidden_states = hidden_gelu * hidden_linear
                    hidden_states = module.dropout(hidden_states)

                    hidden_states = module.wo(hidden_states)
                    return hidden_states

                return forward

            for module in text_encoder.modules():
                if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        t5xxl.to(t5xxl_dtype)
        prepare_fp8(t5xxl.encoder, torch.bfloat16)
        t5xxl.fp8_prepared = True

    # prepare embeddings
    logger.info("Encoding prompts...")
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)

    def encode(prpt: str):
        tokens_and_masks = tokenize_strategy.tokenize(prpt)
        with torch.no_grad():
            if is_fp8(clip_l_dtype):
                with accelerator.autocast():
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, None], tokens_and_masks
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                    l_pooled, _, _, _ = encoding_strategy.encode_tokens(
                        tokenize_strategy, [clip_l, None], tokens_and_masks
                    )

            if is_fp8(t5xxl_dtype):
                with accelerator.autocast():
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        [clip_l, t5xxl],
                        tokens_and_masks,
                        args.apply_t5_attn_mask,
                    )
            else:
                with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                    _, t5_out, txt_ids, t5_attn_mask = encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        [None, t5xxl],
                        tokens_and_masks,
                        args.apply_t5_attn_mask,
                    )
        return l_pooled, t5_out, txt_ids, t5_attn_mask

    l_pooled, t5_out, txt_ids, t5_attn_mask = encode(prompt)
    if negative_prompt:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask = encode(negative_prompt)
    else:
        neg_l_pooled, neg_t5_out, neg_t5_attn_mask = None, None, None

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    # del clip_l, t5xxl
    device_utils.clean_memory()

    # generate image
    logger.info("Generating image...")
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    t5_attn_mask = t5_attn_mask.to(device) if args.apply_t5_attn_mask else None

    x = do_sample(
        accelerator,
        model,
        noise,
        img_ids,
        l_pooled,
        t5_out,
        txt_ids,
        steps,
        guidance,
        t5_attn_mask,
        is_schnell,
        device,
        flux_dtype,
        neg_l_pooled,
        neg_t5_out,
        neg_t5_attn_mask,
        cfg_scale,
        cancel_flag,
        lora_models,
        clora_mode,
        clora_tau,
        mask_interval,
    )
    if x is None:
        return None # canceled 
    if args.offload:
        model = model.cpu()
    # del model
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_latent_height,
        w=packed_latent_width,
        ph=2,
        pw=2,
    )

    # decode
    logger.info("Decoding image...")
    ae = ae.to(device)
    with torch.no_grad():
        if is_fp8(ae_dtype):
            with accelerator.autocast():
                x = ae.decode(x)
        else:
            with torch.autocast(device_type=device.type, dtype=ae_dtype):
                x = ae.decode(x)
    if args.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # save image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{uuid.uuid4()}.png"
    )
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")
    return output_path


def is_fp8(dt):
    return dt in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]


def flux_prepare_generation(
    ckpt_path,
    clip_l_path,
    t5xxl_path,
    ae_path,
    output_dir,
    lora_weights=[],
    offload=None,
    apply_t5_attn_mask=None,
    dtype="bfloat16",
    clip_l_dtype=None,
    ae_dtype=None,
    t5xxl_dtype=None,
    flux_dtype=None,
    merge_lora_weights=None,
):
    device = get_preferred_device()

    def is_fp8(dt):
        return dt in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]

    dtype = str_to_dtype(dtype)
    clip_l_dtype = str_to_dtype(clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(ae_dtype, dtype)
    flux_dtype = str_to_dtype(flux_dtype, dtype)

    logger.info(
        f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}"
    )

    loading_device = "cpu" if offload else device
    use_fp8 = any(
        is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]
    )
    accelerator = accelerate.Accelerator(mixed_precision="bf16") if use_fp8 else None

    logger.info(f"Loading clip_l from {clip_l_path}...")
    clip_l = flux_utils.load_clip_l(clip_l_path, clip_l_dtype, loading_device)
    clip_l.eval()

    logger.info(f"Loading t5xxl from {t5xxl_path}...")
    t5xxl = flux_utils.load_t5xxl(t5xxl_path, t5xxl_dtype, loading_device)
    t5xxl.eval()

    is_schnell, model = flux_utils.load_flow_model(ckpt_path, None, loading_device)
    model.eval()
    model.to(flux_dtype)

    t5xxl_max_length = 256 if is_schnell else 512
    tokenizers_folder = os.path.dirname(clip_l_path)
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(
        t5xxl_max_length, tokenizers_folder
    )
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()
    ae = flux_utils.load_ae(ae_path, ae_dtype, loading_device)

    lora_models: List[lora_flux.LoRANetwork] = []
    for i, weights_file in enumerate(lora_weights or []):
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        is_lora = is_oft = False
        for key in weights_sd.keys():
            if key.startswith("lora"):
                is_lora = True
            if key.startswith("oft"):
                is_oft = True
            if is_lora or is_oft:
                break

        module = oft_flux if is_oft else lora_flux
        lora_model, _ = module.create_network_from_weights(
            multiplier, None, ae, [clip_l, t5xxl], model, weights_sd, True
        )
        lora_model.lora_name = f"lora_{i}"

        if merge_lora_weights:
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        else:
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=False)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)
    fake_args = [
        "--output_dir",
        output_dir,
    ]
    if offload:
        fake_args.append("--offload")
    if apply_t5_attn_mask:
        fake_args.append("--apply_t5_attn_mask")
    # if lora_weights:
    #     fake_args.extend(
    #         ["--lora_weights"]
    #         + (lora_weights if isinstance(lora_weights, list) else [lora_weights])
    #     )
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args(fake_args)
    ae.eval()
    return {
        "model": model,
        "accelerator": accelerator,
        "ae": ae,
        "clip_l": clip_l,
        "t5xxl": t5xxl,
        "device": device,
        "tokenize_strategy": tokenize_strategy,
        "encoding_strategy": encoding_strategy,
        "flux_dtype": flux_dtype,
        "dtype": dtype,
        "device": device,
        "clip_l_dtype": clip_l_dtype,
        "t5xxl_dtype": t5xxl_dtype,
        "is_schnell": is_schnell,
        "flux_dtype": flux_dtype,
        "ae_dtype": ae_dtype,
        "lora_models": lora_models,
        "clora_mode": "soft",
        "clora_tau": 0.07,
        "mask_interval": 1,
        "args": args,
    }


def flux_generate_image_with_prepared_data(prepared_data, cancel_flag):
    return generate_image(
        **prepared_data,
        cancel_flag=cancel_flag,
    )


if __name__ == "__main__":
    target_height = 768  # 1024
    target_width = 1360  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--ae", type=str, required=False)
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, help="directory for caching Tokenizer (for offline inference)")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument(
        "--clip_l_dtype", type=str, default=None, help="dtype for clip_l"
    )
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps. Default is 4 for schnell, 50 for dev",
    )
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--clora_soft", action="store_true", help="Use soft mask (default)")
    parser.add_argument("--clora_hard", action="store_true", help="Use hard argmax mask")
    parser.add_argument("--clora_tau", type=float, default=0.07, help="Temperature for softmax masks")
    parser.add_argument("--mask_interval", type=int, default=1, help="Recompute masks every k steps")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_flux and lora_oft, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument(
        "--merge_lora_weights", action="store_true", help="Merge LoRA weights to model"
    )
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    # Determine CLoRA mode
    clora_mode = "hard" if args.clora_hard else "soft"

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    def is_fp8(dt):
        return dt in [
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]

    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)

    logger.info(
        f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}"
    )

    loading_device = "cpu" if args.offload else device

    use_fp8 = [
        is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]
    ]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()

    # if is_fp8(clip_l_dtype):
    #     clip_l = accelerator.prepare(clip_l)
    # if is_fp8(t5xxl_dtype):
    #     t5xxl = accelerator.prepare(t5xxl)

    # DiT
    is_schnell, model = flux_utils.load_flow_model(args.ckpt_path, None, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(flux_dtype)  # make sure model is dtype
    # if is_fp8(flux_dtype):
    #     model = accelerator.prepare(model)
    #     if args.offload:
    #         model = model.to("cpu")

    t5xxl_max_length = 256 if is_schnell else 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length, args.tokenizer_cache_dir)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # AE
    ae = flux_utils.load_ae(args.ae, ae_dtype, loading_device)
    ae.eval()
    # if is_fp8(ae_dtype):
    #     ae = accelerator.prepare(ae)

    # LoRA
    lora_models: List[lora_flux.LoRANetwork] = []
    for i, weights_file in enumerate(args.lora_weights):
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        is_lora = is_oft = False
        for key in weights_sd.keys():
            if key.startswith("lora"):
                is_lora = True
            if key.startswith("oft"):
                is_oft = True
            if is_lora or is_oft:
                break

        module = lora_flux if is_lora else oft_flux
        lora_model, _ = module.create_network_from_weights(
            multiplier, None, ae, [clip_l, t5xxl], model, weights_sd, True
        )
        lora_model.lora_name = f"lora_{i}"

        if args.merge_lora_weights:
            lora_model.merge_to([clip_l, t5xxl], model, weights_sd)
        else:
            lora_model.apply_to([clip_l, t5xxl], model)
            info = lora_model.load_state_dict(weights_sd, strict=False)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.eval()
            lora_model.to(device)

        lora_models.append(lora_model)

    if not args.interactive:
        generate_image(
            model,
            clip_l,
            t5xxl,
            ae,
            args.prompt,
            args.seed,
            args.width,
            args.height,
            args.steps,
            args.guidance,
            args.negative_prompt,
            args.cfg_scale,
            lora_models=lora_models,
            clora_mode=clora_mode,
            clora_tau=args.clora_tau,
            mask_interval=args.mask_interval,
        )
    else:
        # loop for interactive
        width = target_width
        height = target_height
        steps = None
        guidance = args.guidance
        cfg_scale = args.cfg_scale

        while True:
            print(
                "Enter prompt (empty to exit). Options: --w <width> --h <height> --s <steps> --d <seed> --g <guidance> --m <multipliers for LoRA>"
                " --n <negative prompt>, `-` for empty negative prompt --c <cfg_scale>"
            )
            prompt = input()
            if prompt == "":
                break

            # parse options
            options = prompt.split("--")
            prompt = options[0].strip()
            seed = None
            negative_prompt = None
            for opt in options[1:]:
                try:
                    opt = opt.strip()
                    if opt.startswith("w"):
                        width = int(opt[1:].strip())
                    elif opt.startswith("h"):
                        height = int(opt[1:].strip())
                    elif opt.startswith("s"):
                        steps = int(opt[1:].strip())
                    elif opt.startswith("d"):
                        seed = int(opt[1:].strip())
                    elif opt.startswith("g"):
                        guidance = float(opt[1:].strip())
                    elif opt.startswith("m"):
                        mutipliers = opt[1:].strip().split(",")
                        if len(mutipliers) != len(lora_models):
                            logger.error(
                                f"Invalid number of multipliers, expected {len(lora_models)}"
                            )
                            continue
                        for i, lora_model in enumerate(lora_models):
                            lora_model.set_multiplier(float(mutipliers[i]))
                    elif opt.startswith("n"):
                        negative_prompt = opt[1:].strip()
                        if negative_prompt == "-":
                            negative_prompt = ""
                    elif opt.startswith("c"):
                        cfg_scale = float(opt[1:].strip())
                except ValueError as e:
                    logger.error(f"Invalid option: {opt}, {e}")

            generate_image(
                model,
                clip_l,
                t5xxl,
                ae,
                prompt,
                seed,
                width,
                height,
                steps,
                guidance,
                negative_prompt,
                cfg_scale,
                lora_models=lora_models,
                clora_mode=clora_mode,
                clora_tau=args.clora_tau,
                mask_interval=args.mask_interval,
            )

    logger.info("Done!")
