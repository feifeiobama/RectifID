import torch
from diffusers import DiffusionPipeline


def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    _tmp_sd = pipe.unet.state_dict()
    for key in dW_dict.keys():
        _tmp_sd[key] += dW_dict[key] * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    return pipe

def get_dW_and_merge(pipe_rf, lora_path='Lykon/dreamshaper-7', save_dW = False, base_sd='runwayml/stable-diffusion-v1-5', alpha=1.0):    
    # get weights of base sd models
    _pipe = DiffusionPipeline.from_pretrained(
        base_sd, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    sd_state_dict = _pipe.unet.state_dict()
    
    # get weights of the customized sd models, e.g., the aniverse downloaded from civitai.com    
    _pipe = DiffusionPipeline.from_pretrained(
        lora_path, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    lora_unet_checkpoint = _pipe.unet.state_dict()
    
    # get the dW
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]
    
    # return and save dW dict
    if save_dW:
        save_name = lora_path.split('/')[-1] + '_dW.pt'
        torch.save(dW_dict, save_name)
        
    pipe_rf = merge_dW_to_unet(pipe_rf, dW_dict=dW_dict, alpha=alpha)
    pipe_rf.vae = _pipe.vae
    pipe_rf.text_encoder = _pipe.text_encoder
    
    return dW_dict