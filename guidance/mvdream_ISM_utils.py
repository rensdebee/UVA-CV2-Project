import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler

from diffusers import DDIMScheduler

from .sd_step import *
from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        t_range=[0.02, 0.5],
        max_t_range=0.98,
        guidance_opt=None
    ):
        super().__init__()
        self.guidance_opt = guidance_opt
        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        if guidance_opt.t_range is not None:
            t_range = guidance_opt.t_range

        if guidance_opt.max_t_range is not None:
            max_t_range = guidance_opt.max_t_range

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.dtype = torch.float32

        self.embeddings = {}

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )
        self.sche_func = ddim_step

        self.num_train_timesteps = 1000
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.seed)

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        self.rgb_latent_factors = torch.tensor([
            # R       G       B
            [ 0.298,  0.207,  0.208],
            [ 0.187,  0.286,  0.173],
            [-0.158,  0.189,  0.264],
            [-0.184, -0.271, -0.473]
        ], device=self.device)

    def add_noise_with_cfg(self, latents, noise, 
                           ind_t, ind_prev_t, uncond_text_embedding,
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):

        text_embeddings = text_embeddings.to(self.dtype)

        uncond_text_embedding["context"] = uncond_text_embedding["context"][:4, :, :]
        uncond_text_embedding["camera"] = uncond_text_embedding["camera"][:4, :]

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(self.dtype)
            

            timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1).to(self.device)
            
            # unet_output = unet(cur_noisy_lat_, timestep_model_input, 
            #                     encoder_hidden_states=uncond_text_embedding).sample
            # TODO DEZE LEVERT BULLSHIT OUTPUTS (BLOB IMAGES) OP
            # unet_output = self.model.q_sample(cur_noisy_lat_, timestep_model_input, noise)

            unet_output = self.model.apply_model(cur_noisy_lat_, timestep_model_input, uncond_text_embedding)

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4,1,1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4,1,1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # inverse text (not used... just encodes an empty string)
        self.embeddings["inverse"] = self.encode_text(self.guidance_opt.inverse_text)
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    @torch.no_grad()
    def refine(self, pred_rgb, camera,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)

        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)
            
            tt = torch.cat([t.unsqueeze(0).repeat(batch_size)] * 2).to(self.device)

            noise_pred = self.model.apply_model(latent_model_input, tt, context)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        iteration=0,
        # resolution=(256, 256),        
        # warmup_iter=1500,
        # delta_t=80,
        # delta_t_start = 100,
        # xs_delta_t = 200,
        # xs_inv_steps = 5,
        # denoise_guidance_scale = 1.0,
        # xs_eta = 0.0,
        # grad_scale=.1 # 0.1 in yaml -> 1.0 in init..
    ):
        
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)
        resolution = (256, 256)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, resolution, mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        # camera = convert_opengl_to_blender(camera)
        # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # camera = torch.matmul(flip_yz.to(camera), camera)
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        
        camera = camera.repeat(2, 1)
        # The self.embeddings['neg'] or self.embeddings['pos'] are already repeated copies of the negative or positive prompt 4x 
        # Thus a 4,77,1024 tensor each
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}
        uncond_text_embedding = {"context": torch.cat([self.embeddings["neg"][0].unsqueeze(0), self.embeddings["inverse"]], dim=0).repeat(4,1,1), "camera": camera, "num_frames": 4}

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)

        inverse_text_embeddings = torch.cat([self.embeddings["neg"][0].unsqueeze(0), self.embeddings["inverse"]], dim=0).repeat(4,1,1)

        warm_up_rate = 1. - min(iteration/self.guidance_opt.warmup_iter,1.)

        current_delta_t =  int(self.guidance_opt.delta_t + (warm_up_rate)*(self.guidance_opt.delta_t_start - self.guidance_opt.delta_t))

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        # Step 1: sample x_s with larger steps
        xs_delta_t = self.guidance_opt.xs_delta_t if self.guidance_opt.xs_delta_t is not None else current_delta_t
        xs_inv_steps = self.guidance_opt.xs_inv_steps if self.guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
        starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

        _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, uncond_text_embedding, inverse_text_embeddings, 
                                                                        self.guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=self.guidance_opt.xs_eta)
        # Step 2: sample x_t
        _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, uncond_text_embedding, inverse_text_embeddings, 
                                                                    self.guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)

        pred_scores = pred_scores_xt + pred_scores_xs
        target = pred_scores[0][1]

        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ).to(self.device)
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1).to(self.device)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            unet_output = self.model.apply_model(latent_model_input, tt, context)

            # unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            # noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
            delta_DSD = noise_pred_text - noise_pred_uncond

        pred_noise = noise_pred_uncond + self.guidance_opt.guidance_scale * delta_DSD

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)     

        grad = w(self.alphas[t]) * (pred_noise - target)

        grad = torch.nan_to_num(self.guidance_opt.lambda_guidance * grad)
        loss = SpecifyGradient.apply(latents, grad)

        return loss

        # #TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # if step_ratio is not None:
        #     # dreamtime-like
        #     # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
        #     t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
        #     t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        # else:
        #     t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)

        # # camera = convert_opengl_to_blender(camera)
        # # flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).unsqueeze(0)
        # # camera = torch.matmul(flip_yz.to(camera), camera)
        # # TODO XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        # camera[:, 1] *= -1
        # camera = normalize_camera(camera).view(batch_size, 16)
        
        # camera = camera.repeat(2, 1)
        # # The self.embeddings['neg'] or self.embeddings['pos'] are already repeated copies of the negative or positive prompt 4x 
        # # Thus a 4,77,1024 tensor each
        # embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        # context = {"context": embeddings, "camera": camera, "num_frames": 4}
        # # TODO XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        # # predict the noise residual with unet, NO grad!
        # with torch.no_grad():
        #     # add noise
        #     noise = torch.randn_like(latents)
        #     latents_noisy = self.model.q_sample(latents, t, noise)
        #     # pred noise
        #     latent_model_input = torch.cat([latents_noisy] * 2)
        #     tt = torch.cat([t] * 2)

        #     # import kiui
        #     # kiui.lo(latent_model_input, t, context['context'], context['camera'])
            
        #     noise_pred = self.model.apply_model(latent_model_input, tt, context)

        #     # perform guidance (high scale from paper!)
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # grad = (noise_pred - noise)
        # grad = torch.nan_to_num(grad)

        # # seems important to avoid NaN...
        # # grad = grad.clamp(-1, 1)

        # target = (latents - grad).detach()
        # loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        # return loss
        # #TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

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
        c_ = {"context": self.encode_text(prompts).repeat(4,1,1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4,1,1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(S=num_inference_steps, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc_,
                                        eta=0, x_T=None)

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

        grid = np.concatenate([
            np.concatenate([imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[2], imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()