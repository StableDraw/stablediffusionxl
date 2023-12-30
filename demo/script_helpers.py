import math
import os
import numpy as np
import torch
from typing import List, Union
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast

from ..util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from .discretization import (Img2ImgDiscretizationWrapper, Txt2NoisyDiscretizationWrapper)
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler, DPMPP2SAncestralSampler, EulerAncestralSampler, EulerEDMSampler, HeunEDMSampler, LinearMultistepSampler)
from sgm.util import append_dims, instantiate_from_config

class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)
        
    def __call__(self, image: torch.Tensor):
        """
        Добавляет предопределенную вотермарку к входному изображению

        Args:
            image: ([N], B, C, H, W) in range [0, 1]

        Returns:
            то же, что и на вход, но с вотермаркой
        """
        # Вотермарочная библиотека ожидает вход, как cv2 BGR формат
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n = n)).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image

# Исправленное 48-битное сообщение, которое было рандомно выбрано
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] принимает x как str, использует int, чтобы сконвертировать его в 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watemark = WatermarkEmbedder(WATERMARK_BITS)

def init_st(opt, version_dict, load_ckpt = True, load_filter = True):
    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        if opt["use_custom_ckpt"] == True:
            ckpt = opt["custom_ckpt_name"]
        else:
            ckpt = version_dict["ckpt"]
        config = OmegaConf.load(config)
        model = load_model_from_config(config, ckpt if load_ckpt else None)
        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
        if load_filter:
            state["filter"] = DeepFloydDataFiltering(verbose = opt["verbose"])
    return state

def load_model(model):
    model.cuda()

lowvram_mode = False

def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode

def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()

def load_model_from_config(config, ckpt, verbose = False):
    print(f"Загрузка модели из {ckpt}")
    if ckpt[ckpt.rfind('.'):] == ".safetensors":
        pl_sd = load_safetensors(ckpt, device = "cpu")
    else:
        pl_sd = torch.load(ckpt, map_location = "cpu")
    if "global_step" in pl_sd:
        print(f"Глобальный шаг: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict = False)
    if len(m) > 0 and verbose:
        print("Пропущенные параметры:\n", m)
    if len(u) > 0 and verbose:
        print("Некорректные параматры:")
        print(u)
    model.cuda()
    model.eval()
    return model

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def init_embedder_options(opt, keys, init_dict, prompt = None, negative_prompt = None):
    value_dict = {}
    for key in keys:
        if key == "txt":
            if negative_prompt is None:
                negative_prompt = ""
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt
        if key == "original_size_as_tuple":
            if opt["custom_orig_size"] == True:
                value_dict["orig_width"] = opt["orig_width"]
                value_dict["orig_height"] = opt["orig_height"]
            else:
                value_dict["orig_width"] = init_dict["orig_width"]
                value_dict["orig_height"] = init_dict["orig_height"]
        if key == "crop_coords_top_left":
            value_dict["crop_coords_top"] = opt["crop_coords_top"]
            value_dict["crop_coords_left"] = opt["crop_coords_left"]
        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = opt["aesthetic_score"]
            value_dict["negative_aesthetic_score"] = opt["negative_aesthetic_score"]
        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]
    return value_dict

def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watemark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{base_count:09}.png"))
        base_count += 1

def get_guider(opt):
    guider = opt["guider_discretization"]
    if guider == "IdentityGuider":
        guider_config = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = opt["guidance_scale"]
        dyn_thresh_config = {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"}
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config}
        }
    return guider_config

def init_sampling(opt, img2img_strength = 1.0, specify_num_samples = True, stage2strength = None):
    num_rows, num_cols = 1, 1
    if specify_num_samples:
        num_cols = opt["num_cols"]
    discretization = opt["sampling_discretization"]
    discretization_config = get_discretization(opt, discretization)
    guider_config = get_guider(opt)
    sampler = get_sampler(opt, discretization_config, guider_config)
    if img2img_strength < 1.0:
        print(f"Предупреждение: {sampler.__class__.__name__} с Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(sampler.discretization, strength = img2img_strength)
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(sampler.discretization, strength = stage2strength, original_steps = opt["steps"])
    return sampler, num_rows, num_cols

def get_discretization(opt, discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {"target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"}
    elif discretization == "EDMDiscretization":
        sigma_min = opt["sigma_min"]
        sigma_max = opt["sigma_max"]
        rho = opt["rho"]
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho
            }
        }
    return discretization_config

def get_sampler(opt, discretization_config, guider_config):
    sampler_name = opt["sampler"]
    steps = opt["steps"]
    s_churn = (opt["s_churn"] * steps * (2 ** 0.5 - 1))
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, s_churn = s_churn, s_tmin = opt["s_tmin"], s_tmax = opt["s_tmax"], s_noise = opt["s_noise"], verbose = opt["verbose"])
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, s_churn = s_churn, s_tmin = opt["s_tmin"], s_tmax = opt["s_tmax"], s_noise = opt["s_noise"], verbose = opt["verbose"])
    elif (sampler_name == "EulerAncestralSampler" or sampler_name == "DPMPP2SAncestralSampler"):
        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, eta = opt["eta"], s_noise = opt["s_noise"], verbose = opt["verbose"])
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, eta = opt["eta"], s_noise = opt["s_noise"], verbose = opt["verbose"])
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, verbose = opt["verbose"])
    elif sampler_name == "LinearMultistepSampler":
        order = opt["order"]
        sampler = LinearMultistepSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, order = order, verbose = opt["verbose"])
    return sampler

def do_text2img(model, sampler, value_dict, num_samples, H, W, C, F, force_uc_zero_embeddings: List = None, batch2model_input: List = None, return_latents = False, filter = None):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []
    print("Обработка")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                model.conditioner.cuda()
                batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples)
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc = batch_uc, force_uc_zero_embeddings = force_uc_zero_embeddings)
                unload_model(model.conditioner)
                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]
                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)
                model.denoiser.cuda()
                model.model.cuda()
                samples_z = sampler(denoiser, randn, cond = c, uc = uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                model.first_stage_model.cuda()
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min = 0.0, max = 1.0)
                unload_model(model.first_stage_model)
                if filter is not None:
                    samples = filter(samples)
                if return_latents:
                    return samples, samples_z
                return samples

def get_batch(keys, value_dict, N: Union[List, ListConfig], device = "cuda"):
    # Захардкоженные демонстрационные пресеты
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "txt":
            batch["txt"] = (np.repeat([value_dict["prompt"]], repeats = math.prod(N)).reshape(N).tolist())
            batch_uc["txt"] = (np.repeat([value_dict["negative_prompt"]], repeats = math.prod(N)).reshape(N).tolist())
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1))
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1))
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1))
            batch_uc["aesthetic_score"] = (torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1))
        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1))
        else:
            batch[key] = value_dict[key]
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

@torch.no_grad()
def do_img2img(img, model, sampler, value_dict, num_samples, force_uc_zero_embeddings = [], additional_kwargs = {}, offset_noise_level: int = 0.0, return_latents = False, skip_encode = False, filter = None, add_noise = True):
    print("Обработка")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                model.conditioner.cuda()
                batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [num_samples])
                c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc = batch_uc, force_uc_zero_embeddings = force_uc_zero_embeddings)
                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to("cuda"), (c, uc))
                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    model.first_stage_model.cuda()
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)
                noise = torch.randn_like(z)
                sigmas = sampler.discretization(sampler.num_steps).cuda()
                sigma = sigmas[0]
                print(f"Все сигмы: {sigmas}")
                print(f"Шумовая сигма: {sigma}")
                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device = z.device), z.ndim)
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                    noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)  # Заметка: захардкожено в DDPM-подобное преобразование. 
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)
                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)
                model.denoiser.cuda()
                model.model.cuda()
                samples_z = sampler(denoiser, noised_z, cond = c, uc = uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                model.first_stage_model.cuda()
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                if filter is not None:
                    samples = filter(samples)
                if return_latents:
                    return samples, samples_z
                return samples