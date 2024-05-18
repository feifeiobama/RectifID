import os
import os.path as osp
import time
import argparse
import copy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils.torch_utils import randn_tensor
from piecewise_rectified_flow.src.scheduler_perflow import PeRFlowScheduler
from InstaFlow.code.pipeline_rf import RectifiedFlowPipeline
from utils import get_dW_and_merge

import insightface
import onnxruntime as ort
from onnx2torch import convert
import kornia
import tensorflow as tf
from deepface import DeepFace


class PersonRFlow:
    def __init__(self, model, seed, device):
        self.seed = seed
        self.device = device

        if model == '2RFlow':
            self.pipe = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", safety_checker=None, torch_dtype=torch.float16)
            get_dW_and_merge(self.pipe, lora_path="Lykon/dreamshaper-7", save_dW=False, alpha=1.0)
            self.guidance_scale = 1.5
        elif 'PeRFlow' in model:
            # the paper only uses SD 1.5 while it also supports SD 2.1
            self.pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd15-dreamshaper", safety_checker=None, torch_dtype=torch.float16)
            self.pipe.scheduler = PeRFlowScheduler.from_config(self.pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
            # self.pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd21-artius", safety_checker=None, torch_dtype=torch.float16)
            # self.pipe.scheduler = PeRFlowScheduler.from_config(self.pipe.scheduler.config, prediction_type="velocity", num_time_windows=4)
            self.guidance_scale = 3.0
            # self.guidance_scale = 4.5
        else:
            raise Exception('RFlow type %s not implemented' % model)
        self.size = 512
        # self.size = 768
        
        # if model == 'PeRFlow-IPAdapter':
        #     self.enable_ipadapter()
        # else:
        #     self.disable_ipadapter()
        self.ipadapter = False

        for module in [self.pipe.vae, self.pipe.text_encoder, self.pipe.unet]:
            for param in module.parameters():
                param.requires_grad = False

        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self._forward = self.pipe.__call__.__wrapped__

        # antelopev2
        # https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo
        self.detector = insightface.model_zoo.get_model('scrfd_10g_bnkps.onnx', provider_options=[{'device_id': self.device}, {}])
        self.detector.prepare(ctx_id=0, input_size=(640, 640))
        self.model = convert('glintr100.onnx').eval().to('cuda')
        for param in self.model.parameters():
            param.requires_grad_(False)

        # for deepface
        tf.config.set_visible_devices([], device_type='GPU')
    
    # def enable_ipadapter(self, scale=0.5):
    #     self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
    #     self.pipe.set_ip_adapter_scale(scale)
    #     self.ipadapter = True
    
    # def disable_ipadapter(self):
    #     self.pipe.unload_ip_adapter()
    #     self.ipadapter = False

    def prepare(self, ref):
        ref_image = Image.open(ref).convert("RGB")

        with torch.no_grad():
            det_thresh_backup = self.detector.det_thresh
            boxes = []
            while len(boxes) == 0:
                boxes, kpss = self.detector.detect(np.array(ref_image), max_num=1)
                self.detector.det_thresh -= 0.1
            self.detector.det_thresh = det_thresh_backup
            M = insightface.utils.face_align.estimate_norm(kpss[0])
            ref_image_cropped = kornia.geometry.transform.warp_affine(
                TF.to_tensor(ref_image).unsqueeze(0).to('cuda'), torch.tensor(M).float().unsqueeze(0).to('cuda'), (112, 112)
            ) * 2 - 1

            self.ref_embedding = self.model(ref_image_cropped)
            self.cropped_image = np.array((ref_image_cropped[0] / 2 + 0.5).cpu().permute(1, 2, 0) * 255, dtype=np.uint8)
        
        try:
            self.attribute = DeepFace.analyze(img_path=ref, actions = ['gender', 'race'])
        except:
            self.attribute = None
    
    def forward(self, prompt, num_steps, latents0, callback):
        if not self.ipadapter:
            return self._forward(self.pipe, prompt=prompt, height=self.size, width=self.size, num_inference_steps=num_steps, guidance_scale=self.guidance_scale,
                latents=latents0, output_type='pt', return_dict=False, callback_on_step_end=callback)[0][0]
        else:
            return self._forward(self.pipe, prompt=prompt, height=self.size, width=self.size, ip_adapter_image=self.cropped_image, num_inference_steps=num_steps,
                guidance_scale=self.guidance_scale, latents=latents0, output_type='pt', return_dict=False, callback_on_step_end=callback)[0][0]
    
    def generate(self, prompt, seed, out, num_iterations=50, num_steps=4, verbose=True, guidance=1.):
        if self.attribute is not None:
            idx = np.argmax([a['region']['w'] * a['region']['h'] for a in self.attribute])
            if self.attribute[idx]['dominant_gender'] == 'Man':
                prompt = prompt.replace('person', self.attribute[idx]['dominant_race'] + ' man')
            else:
                prompt = prompt.replace('person', self.attribute[idx]['dominant_race'] + ' woman')
        prompt = prompt + ', face'

        generator = torch.manual_seed(seed)
        latent_size = int(self.size // 8)
        latents = nn.Parameter(randn_tensor((num_steps, 4, latent_size, latent_size), generator=generator, device=self.pipe._execution_device, dtype=self.pipe.text_encoder.dtype))
        latents0 = latents[:1].data.clone()
        optimizer = torch.optim.SGD([latents], guidance)

        iterations = range(num_iterations)
        if verbose:
            iterations = tqdm(iterations)
        
        latents_last = latents.data.clone()
        latents_last_e = latents.data.clone()
        initialized_i = -1

        def callback(self, i, t, callback_kwargs):
            nonlocal latents_last, latents_last_e, initialized_i
            if initialized_i < i:
                latents[i:(i+1)].data.copy_(callback_kwargs['latents'])
                latents_last[i:(i+1)].copy_(callback_kwargs['latents'])
                latents_last_e[i:(i+1)].copy_(callback_kwargs['latents'])
                initialized_i = i
            if i < num_steps - 1:
                callback_kwargs['latents'] += latents[(i+1):(i+2)] - latents[(i+1):(i+2)].detach()
            latents_e = callback_kwargs['latents'].data.clone()
            callback_kwargs['latents'] += latents_last[i:(i+1)].detach() - callback_kwargs['latents'].detach()
            callback_kwargs['latents'] += latents_e.detach() - latents_last_e[i:(i+1)].detach()
            # callback_kwargs['latents'] += latents[i:(i+1)].detach() - latents_last_e[i:(i+1)].detach()
            gamma_s_e = self.scheduler.get_window_alpha(t)[4]
            callback_kwargs['latents'] += (latents[i:(i+1)].detach() - latents_last_e[i:(i+1)].detach()) * ((1 - gamma_s_e ** 2) ** 0.5)
            latents_last[i:(i+1)].copy_(callback_kwargs['latents'])
            latents_last_e[i:(i+1)].data.copy_(latents_e)
            latents[i:(i+1)].data.copy_(latents_e)
            return callback_kwargs

        for iteration in iterations:
            image = self.forward(prompt=prompt, num_steps=num_steps, latents0=latents0+latents[:1]-latents[:1].detach(), callback=callback)
            
            det_thresh_backup = self.detector.det_thresh
            boxes = []
            while len(boxes) == 0:
                boxes, kpss = self.detector.detect(np.array(image.data.permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8), max_num=1)
                self.detector.det_thresh -= 0.1
            self.detector.det_thresh = det_thresh_backup
            
            M = insightface.utils.face_align.estimate_norm(kpss[0])
            image_cropped = kornia.geometry.transform.warp_affine(
                image.float().unsqueeze(0), torch.tensor(M).float().unsqueeze(0).to('cuda'), (112, 112)
            ) * 2 - 1
            embedding = self.model(image_cropped)
            loss = (1 - F.cosine_similarity(embedding, self.ref_embedding)) * 100

            optimizer.zero_grad()
            loss.backward()
            grad_norm = latents.grad.reshape(num_steps, -1).norm(dim=-1)
            latents.grad /= grad_norm.reshape(num_steps, 1, 1, 1).clamp(min=1)
            optimizer.step()
            del image

        with torch.no_grad():
            image = self.forward(prompt=prompt, num_steps=num_steps, latents0=latents0, callback=callback)
        os.makedirs(osp.dirname(out), exist_ok=True)
        plt.imsave(out, np.array(image.permute(1, 2, 0).cpu() * 255, dtype=np.uint8))
    
    def generate_multi_prompt(self, prompts, outs, num_steps=4):
        for prompt, out in zip(prompts, outs):
            self.generate(prompt, self.seed, out, num_steps=num_steps)


def parse_args(single_ref=False):
    parser = argparse.ArgumentParser()
    if single_ref:
        parser.add_argument('--ref', default='assets/hinton.jpg', type=str)
        parser.add_argument('--prompt', default=[], type=str, nargs='+')
        parser.add_argument('--out', default=[], type=str, nargs='+')
    parser.add_argument('--model', default='PeRFlow', type=str)
    parser.add_argument('--iteration', default=50, type=int)
    parser.add_argument('--step', default=4, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--guidance', default=1., type=float)
    args = parser.parse_args()

    if single_ref:
        assert len(args.prompt) == len(args.out)
        
    # os.environ['http_proxy'] = "" 
    # os.environ['https_proxy'] = ""
    torch.cuda.set_device('cuda:%d' % args.device)
    
    return args


def main():
    args = parse_args(single_ref=True)
    model = PersonRFlow(args.model, args.seed, args.device)
    model.prepare(args.ref)
    model.generate_multi_prompt(args.prompt, args.out, num_iterations=args.iteration, num_steps=args.step, guidance=args.guidance)


if __name__ == '__main__':
    main()
    