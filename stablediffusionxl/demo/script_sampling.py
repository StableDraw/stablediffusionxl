from pytorch_lightning import seed_everything
from PIL import Image
from io import BytesIO
from .script_helpers import *

SAVE_PATH = "outputs/demo/txt2img/"

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576)
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "is_legacy": False,
        "config": "configs/sd_xl_base.yaml",
        "ckpt": "weights/sd_xl_base_1.0.safetensors"
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "is_legacy": False,
        "config": "configs/sd_xl_base.yaml",
        "ckpt": "weights/sd_xl_base_0.9.safetensors"
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "is_legacy": True,
        "config": "configs/sd_2_1.yaml",
        "ckpt": "weights/v2-1_512-ema-pruned.safetensors"
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "is_legacy": True,
        "config": "configs/sd_2_1_768.yaml",
        "ckpt": "weights/v2-1_768-ema-pruned.safetensors"
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "is_legacy": True,
        "config": "configs/sd_xl_refiner.yaml",
        "ckpt": "weights/sd_xl_refiner_0.9.safetensors"
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "is_legacy": True,
        "config": "configs/sd_xl_refiner.yaml",
        "ckpt": "weights/sd_xl_refiner_1.0.safetensors"
    },
}

def load_img(binary_data, max_dim):
    image = Image.open(BytesIO(binary_data)).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Загружено входное изображение размера ({orig_w}, {orig_h})")
    cur_dim = orig_w * orig_h
    if cur_dim > max_dim:
        k = cur_dim / max_dim
        sk = float(k ** (0.5))
        w, h = int(orig_w / sk), int(orig_h / sk)
    else:
        w, h = orig_w, orig_h
    w, h = map(lambda x: x - x % 64, (w, h))  # изменение размера в целое число, кратное 64-м
    if w == 0 and orig_w != 0:
        w = 64
    if h == 0 and orig_h != 0:
        h = 64
    if (w, h) != (orig_w, orig_h):
        image = image.resize((w, h), resample = Image.LANCZOS)
        print(f"Размер изображения изменён на ({w}, {h} (w, h))")
    else:
        print(f"Размер исходного изображения не был изменён")
    return image

def prepare(opt):
    version_dict = VERSION2SPECS[opt["version"]]
    set_lowvram_mode(opt["low_vram_mode"])
    seed_everything(opt["seed"])
    state = init_st(opt = opt, version_dict = version_dict, load_filter = opt["use_filter"])
    is_legacy = version_dict["is_legacy"]
    if is_legacy:
        negative_prompt = opt["negative_prompt"]
    else:
        negative_prompt = ""  #оно не используется
    stage2strength = None
    finish_denoising = False
    state2 = None
    sampler2 = None
    if opt["version"].startswith("SDXL-base") and opt["refiner"] != "":
        add_pipeline = True
        state2 = init_st(opt = opt, version_dict = VERSION2SPECS[opt["refiner"]], load_filter = False)
        stage2strength = opt["refinement_strength"]
        sampler2, *_ = init_sampling(opt = opt, img2img_strength = stage2strength, specify_num_samples = False)
        finish_denoising = opt["finish_denoising"]
        if not finish_denoising:
            stage2strength = None
    else:
        add_pipeline = False
    return state, add_pipeline, stage2strength, state2, sampler2, finish_denoising, negative_prompt

def postprocessing(opt, prompt, state, finish_denoising, out, add_pipeline, state2, sampler2):
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None
    if add_pipeline and samples_z is not None:
        print("Запуск этапа уточнения")
        samples = apply_refiner(opt, samples_z, state2, sampler2, samples_z.shape[0], prompt = prompt, filter = state.get("filter"), finish_denoising = finish_denoising)
    if opt["add_watermark"] == True:
        samples = embed_watemark(samples)
    r = []
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        image = Image.fromarray(sample.astype(np.uint8))
        buf = BytesIO()
        image.save(buf, format = "PNG")
        b_data = buf.getvalue()
        image.close
        r.append(b_data)
    torch.cuda.empty_cache()
    return r

def apply_refiner(opt, input, state, sampler, num_samples, prompt, filter = None, finish_denoising = False):
    version_dict = VERSION2SPECS[opt["version"]]
    init_dict = {"orig_width": input.shape[3] * opt["m_k"], "orig_height": input.shape[2] * opt["m_k"], "target_width": input.shape[3] * opt["m_k"], "target_height": input.shape[2] * opt["m_k"]}
    value_dict = init_dict
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = opt["negative_prompt"] if version_dict["is_legacy"] else ""
    value_dict["crop_coords_top"] = opt["crop_coords_top"]
    value_dict["crop_coords_left"] = opt["crop_coords_left"]
    value_dict["aesthetic_score"] = opt["aesthetic_score"]
    value_dict["negative_aesthetic_score"] = opt["negative_aesthetic_score"]
    print(f"Пропорции входного изображения рефайнера: {input.shape}")
    samples = do_img2img(input, state["model"], sampler, value_dict, num_samples, skip_encode = True, filter = filter, add_noise = not finish_denoising)
    return samples