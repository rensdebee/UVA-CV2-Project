import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros(
            [1], device=input_tensor.device, dtype=input_tensor.dtype
        )  # dummy loss value

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        (gt_grad,) = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name="sd-v2.1-base-4view",
        ckpt_path=None,
        t_range=[0.02, 0.98],
        guidance_opt=None,
    ):
        super().__init__()
        self.guidance_opt = guidance_opt

        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = (
            build_model(self.model_name, ckpt_path=self.ckpt_path)
            .eval()
            .to(self.device)
        )
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.embeddings = {}
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder"
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="scheduler",
            torch_dtype=self.dtype,
        )
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4, 1, 1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4, 1, 1)
        self.embeddings["pos"] = pos_embeds
        self.embeddings["neg"] = neg_embeds
        self.embeddings["inverse"] = self.encode_text(self.guidance_opt.inverse_text)
        from guidance.lora_unet import UNet2DConditionModel
        from diffusers.loaders import AttnProcsLayers
        from diffusers.models.attention_processor import LoRAAttnProcessor
        import einops

        _unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None,
        ).to(self.device)
        _unet.requires_grad_(False)
        lora_attn_procs = {}
        for name in _unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else _unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = _unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = _unet.config.block_out_channels[block_id]
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
        _unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(_unet.attn_processors)
        device = self.device
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        class LoraUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = _unet
                self.sample_size = 64
                self.in_channels = 4
                self.device = device
                self.dtype = torch.float32
                self.text_embeddings = text_embeddings

            def forward(self, x, t, c=None, shading="albedo"):
                textemb = einops.repeat(
                    self.text_embeddings, "1 L D -> B L D", B=x.shape[0]
                ).to(device)
                return self.unet(
                    x, t, encoder_hidden_states=textemb, c=c, shading=shading
                )

        self._unet = _unet
        self.lora_layers = lora_layers
        self.unet = LoraUnet().to(device)

    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings

    @torch.no_grad()
    def refine(
        self,
        pred_rgb,
        camera,
        guidance_scale=100,
        steps=50,
        strength=0.8,
    ):

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(
            pred_rgb, (256, 256), mode="bilinear", align_corners=False
        )
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(
            latents, torch.randn_like(latents), self.scheduler.timesteps[init_step]
        )

        camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)

        embeddings = torch.cat(
            [
                self.embeddings["neg"].repeat(real_batch_size, 1, 1),
                self.embeddings["pos"].repeat(real_batch_size, 1, 1),
            ],
            dim=0,
        )
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):

            latent_model_input = torch.cat([latents] * 2)

            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,  # [B, C, H, W], B is multiples of 4
        camera,  # [B, 4, 4]
        shading="albedo",
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
    ):
        q_unet = self.unet
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = (
                F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(
                pred_rgb, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(
                self.min_step, self.max_step
            )
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (real_batch_size,),
                dtype=torch.long,
                device=self.device,
            ).repeat(4)

        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        poses = camera.view(batch_size, 16)
        camera = camera[:, [0, 2, 1, 3]]  # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        camera = camera.repeat(2, 1)
        embeddings = torch.cat(
            [
                self.embeddings["neg"].repeat(real_batch_size, 1, 1),
                self.embeddings["pos"].repeat(real_batch_size, 1, 1),
            ],
            dim=0,
        )
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # import kiui
            # kiui.lo(latent_model_input, t, context['context'], context['camera'])

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

        noise_pred_q = q_unet(latents_noisy, t, c=poses, shading=shading).sample

        sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (
            1 - self.scheduler.alphas_cumprod.to(self.device)[t]
        ) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_pred_q = (
            sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy
        )

        grad = noise_pred - noise_pred_q
        grad = torch.nan_to_num(grad)

        loss = SpecifyGradient.apply(latents, grad)

        return loss, latents, poses

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        batch_size = len(prompts) * 4

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]
        c_ = {"context": self.encode_text(prompts).repeat(4, 1, 1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4, 1, 1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(
            S=num_inference_steps,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc_,
            eta=0,
            x_T=None,
        )

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = MVDream(device)

    while True:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate(
            [
                np.concatenate([imgs[0], imgs[1]], axis=1),
                np.concatenate([imgs[2], imgs[3]], axis=1),
            ],
            axis=0,
        )

        # visualize image
        plt.imshow(grid)
        plt.show()
