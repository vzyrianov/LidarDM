from typing import Any, Dict
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, PNDMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, SchedulerMixin
from .gumbel_sigmoid import gumbel_sigmoid

__all__ = ["DiffusionPipelineFieldCond", "DiffusionPipelineField", "DiffusionPipelineLiDARGenSequence", "ScheduleConfig"]


@dataclass
class ScheduleConfig:
    second_to_last_noise_samples: int = 2
    last_noise_samples: int = 6

    use_noise_second_to_last: bool = True
    use_noise_last: bool = False


class DiffusionPipelineBase(nn.Module):
    def __init__(self, autoencoder, unet) -> None:
        super().__init__()

        self.vae = autoencoder
        self.unet = unet
        self.vae.eval()

        self.scheduler = PNDMScheduler(num_train_timesteps=1000,
                                       beta_start=0.00085,
                                       beta_end=0.012,
                                       beta_schedule='scaled_linear')
        
    def schedule_cleanup_settings(self, i, scheduler, schedule_config):
        skip_noise = False
        append_now = False

        if (i < scheduler.timesteps.shape[0]-1):
            t = scheduler.timesteps[i]

            if((scheduler.timesteps.shape[0]-1) - i <15):
                append_now=True
        else:
            append_now=True
            i_new = (i+1) - scheduler.timesteps.shape[0]

            if(i_new < schedule_config.second_to_last_noise_samples):
                t = scheduler.timesteps[-2]
                skip_noise = not schedule_config.use_noise_second_to_last
            else:
                t = scheduler.timesteps[-1]
                skip_noise = (not schedule_config.use_noise_last) or ((i_new+1) == schedule_config.second_to_last_noise_samples + schedule_config.last_noise_samples)
        
        return t, skip_noise, append_now

class DiffusionPipelineLiDARGenSequence(DiffusionPipelineBase):

    def __init__(self, map_autoencoder=None, latent_size=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.is_cond = False

        if(map_autoencoder is not None):
            self.latent_size = latent_size
            self.is_cond = True
            self.map_vae = map_autoencoder
            self.map_vae.eval()

            for param in self.map_vae.parameters():
                param.requires_grad=False
        else:
            self.latent_size=4


    def forward_train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x = data["lidar_seq"]
        batch_size = x.shape[0]
        device = x.device

        if(self.is_cond):
            with torch.no_grad():
                cond = self.map_vae.encode(data["bev"]).latent_dist.sample()

        with torch.no_grad():

            if(self.is_cond):

                #Drop label 20% of the time
                do_cond = torch.zeros((batch_size)).bernoulli_(0.8).to(device)
                do_cond_multiplier = torch.transpose(do_cond.repeat((40,4,40,1)), 0, 3)
                cond = cond * do_cond_multiplier
            
            latent_individual = []
            for i in range(0, 5):
                latent_individual.append(
                    self.vae.encode(data["lidar_seq"][:,i]).latent_dist.sample() * self.vae.config.scaling_factor
                )

            latent_stack = torch.stack(latent_individual, dim = 1)
            latent = latent_stack.reshape((
                    latent_stack.shape[0],
                    #Along dim 1: [f1c1, f1c2, f1c3, f1c4, f2c1, ...]
                    latent_stack.shape[1]*latent_stack.shape[2],
                    latent_stack.shape[3],
                    latent_stack.shape[4])) 
            

            noise = torch.randn_like(latent)

            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.long,
                device=device,
            )

            noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

        if(self.is_cond):
            data["diffusion/pred_noise"] = self.unet(noisy_latent, cond, timesteps).sample
        else:
            data["diffusion/pred_noise"] = self.unet(noisy_latent, timesteps).sample

        data["lidar/latent"] = latent
        data["diffusion/target_noise"] = noise
        return data
    
    def decode_stack_of_latents(self, latents):
        latents_s = latents / self.vae.config.scaling_factor
        decoded = []
        for i in range(0, latents.shape[1] // self.latent_size):
            x = self.vae.decode(latents_s[:,0+(i*self.latent_size):((i+1)*self.latent_size)]).sample
            decoded.append(x)
        
        return decoded

    @torch.no_grad()
    def forward_test(self,
                     data: Dict[str, Any], 
                     use_gumbel: bool = False,
                     skip_act_and_threshold: bool = False,
                     num_steps: int = 50,
                     scheduler: SchedulerMixin = None,
                     threshold: float = 0.8,
                     gumbel_tau: float = 2.0,
                     schedule_config: ScheduleConfig = ScheduleConfig(),
                     guidance_scale=4.0) -> Dict[str, Any]:
        batch_size = data["lidar"].shape[0]
        
        if(self.is_cond):
            with torch.no_grad():
                cond = self.map_vae.encode(data["bev"]).latent_dist.sample()
            latents = self.sample_model_cond(cond, guidance_scale, num_steps, device=data["lidar"].device, scheduler=self.scheduler if (scheduler is None) else scheduler, generate_list=False, schedule_config=schedule_config)

        else:
            latents = self.sample_model(batch_size, num_steps, device=data["lidar"].device, scheduler=self.scheduler if (scheduler is None) else scheduler, generate_list=False, schedule_config=schedule_config)

        decoded = self.decode_stack_of_latents(latents)


        current_dtype = decoded[0].dtype

        if(not skip_act_and_threshold):
            if use_gumbel:
                decoded = [(gumbel_sigmoid(decoded, 1.0, 0.0, tau=gumbel_tau, threshold=threshold, hard=True)).to(current_dtype) for d in decoded]
            else:
                decoded = [(torch.nn.functional.sigmoid(d) > 0.5).to(current_dtype) for d in decoded]

        data["sample"] = decoded[0]
        data["sample_seq"] = decoded

        return data



    def sample_model_cond(self, cond, guidance_scale, sampling_steps,  device, scheduler, generate_list=False, schedule_config=ScheduleConfig()):

        generated_list = []

        batch_size = cond.shape[0]

        do_classifier_guidance = True
        cond_input = torch.cat([cond*0.0, cond]) if do_classifier_guidance else cond

        scheduler_timesteps_copy = scheduler.config.num_train_timesteps
        
        latent_size = self.latent_size

        latents = torch.randn((batch_size, (latent_size*5),
                     40,
                     40)).to(device)


        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(sampling_steps)

        for i in range(scheduler.timesteps.shape[0] + schedule_config.second_to_last_noise_samples + schedule_config.last_noise_samples):

            t, skip_noise, append_now = self.schedule_cleanup_settings(i, scheduler, schedule_config)
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_guidance else latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_prediction = self.unet(latent_model_input, cond_input, t).sample

            if(do_classifier_guidance):
                
                noise_pred_uncond, noise_pred_cond = noise_prediction.chunk(2)
                noise_prediction = noise_pred_uncond  + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                #TODO: Scale? https://github.com/huggingface/diffusers/blob/58f5f748f4c8f63db787238392738ba3b377dbe4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L692C1-L692C115
                #noise_prediction = self.rescale_noise_cfg(noise_prediction, noise_pred_cond)
            
            if(self.scheduler is scheduler):
                latents = scheduler.step(noise_prediction, t, latents).prev_sample
            else:
                latents = scheduler.step(noise_prediction, t, latents, skip_noise=skip_noise).prev_sample

            if(generate_list and append_now):
                generated_list.append(latents)
        
        scheduler.set_timesteps(scheduler_timesteps_copy)

        if(generate_list):
            return generated_list

        return latents

    def sample_model(self, batch_size, sampling_steps, device, scheduler, generate_list=False, schedule_config=ScheduleConfig()):
        
        generated_list = []

        scheduler_timesteps_copy = scheduler.config.num_train_timesteps

        latents = torch.randn((batch_size, 5*4,
                     80,
                     80)).to(device)

        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(sampling_steps)

        #for t in scheduler.timesteps:
        for i in range(scheduler.timesteps.shape[0] + schedule_config.second_to_last_noise_samples + schedule_config.last_noise_samples):

            t, skip_noise, append_now = self.schedule_cleanup_settings(i, scheduler, schedule_config)


            latent_model_input = scheduler.scale_model_input(latents, t)

            noise_prediction = self.unet(latent_model_input, t).sample
            if(self.scheduler is scheduler):
                latents = scheduler.step(noise_prediction, t, latents).prev_sample
            else:
                latents = scheduler.step(noise_prediction, t, latents, skip_noise=skip_noise).prev_sample

            if(generate_list and append_now):
                generated_list.append(latents)
        
        scheduler.set_timesteps(scheduler_timesteps_copy)

        if(generate_list):
            return generated_list

        return latents

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_test(data)

class DiffusionPipelineField(DiffusionPipelineBase):

    def __init__(self, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = channels

    def forward_train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        #x_lidar = data["lidar"]
        x_field = data["field"]

        batch_size = x_field.shape[0]
        device = x_field.device

        with torch.no_grad():
            latent = self.vae.encode(x_field).latent_dist.sample() * self.vae.config.scaling_factor
            #latent = self.vae.encode(data["lidar"]).latent_dist.sample() * self.vae.config.scaling_factor
            
            noise = torch.randn_like(latent)

            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.long,
                device=device,
            )

            noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

        data["diffusion/pred_noise"] = self.unet(noisy_latent, timesteps).sample
        data["lidar/latent"] = latent
        data["diffusion/target_noise"] = noise
        return data

    @torch.no_grad()
    def forward_test(self,
                     data: Dict[str, Any],
                     use_gumbel: bool = False,
                     skip_act_and_threshold: bool = False,
                     num_steps: int = 50,
                     scheduler: SchedulerMixin = None,
                     threshold: float = 0.8,
                     gumbel_tau: float = 2.0,
                     generate_video: bool = False,
                     schedule_config: ScheduleConfig = ScheduleConfig()) -> Dict[str, Any]:
        batch_size = data["lidar"].shape[0]
        
        latents = self.sample_model(batch_size, num_steps, device=data["lidar"].device, scheduler=self.scheduler if (scheduler is None) else scheduler, generate_list=generate_video, schedule_config=schedule_config)

        decoded = [self.vae.decode(l / self.vae.config.scaling_factor).sample for l in latents]

        current_dtype = decoded[0].dtype
        
        decoded_field = [d[:].detach().cpu() for d in decoded]
        
        decoded_field = [torch.nn.functional.sigmoid(d) for d in decoded_field]

          
        #data["sample"] = decoded_lidar[-1]
        data["sample_field"] = decoded_field[-1]

        data["video_field"] = decoded_field

        return data

    def sample_model(self, batch_size, sampling_steps, device, scheduler, generate_list=False, schedule_config=ScheduleConfig()):
        
        generated_list = []

        scheduler_timesteps_copy = scheduler.config.num_train_timesteps
        
        latent_size = self.channels
        latents = torch.randn((batch_size, latent_size,
                     40,
                     40)).to(device)

        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(sampling_steps)

        #for t in scheduler.timesteps:
        for i in range(scheduler.timesteps.shape[0] + schedule_config.second_to_last_noise_samples + schedule_config.last_noise_samples):

            t, skip_noise, append_now = self.schedule_cleanup_settings(i, scheduler, schedule_config)


            latent_model_input = scheduler.scale_model_input(latents, t)

            noise_prediction = self.unet(latent_model_input, t).sample
  
            if(self.scheduler is scheduler):
                latents = scheduler.step(noise_prediction, t, latents).prev_sample
            else:
                latents = scheduler.step(noise_prediction, t, latents, skip_noise=skip_noise).prev_sample

            if(generate_list and append_now):
                generated_list.append(latents)
        
        scheduler.set_timesteps(scheduler_timesteps_copy)

        if(generate_list):
            return generated_list

        return [latents]
        

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_test(data)
        


class DiffusionPipelineFieldCond(DiffusionPipelineBase):

    def __init__(self, map_autoencoder, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.map_vae = map_autoencoder
        self.map_vae.eval()

        for param in self.map_vae.parameters():
            param.requires_grad=False

        self.channels = channels

    def forward_train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        x_field = data["field"]
        batch_size = x_field.shape[0]
        device = x_field.device

        with torch.no_grad():
            cond = self.map_vae.encode(data["bev"]).latent_dist.sample()

        with torch.no_grad():
            #Drop label 20% of the time
            do_cond = torch.zeros((batch_size)).bernoulli_(0.8).to(device)
            do_cond_multiplier = torch.transpose(do_cond.repeat((40,4,40,1)), 0, 3)
            cond = cond * do_cond_multiplier

            latent = self.vae.encode(x_field).latent_dist.sample() * self.vae.config.scaling_factor
            
            noise = torch.randn_like(latent)

            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.long,
                device=device,
            )

            noisy_latent = self.scheduler.add_noise(latent, noise, timesteps)

        data["diffusion/pred_noise"] = self.unet(noisy_latent, cond, timesteps).sample
        data["lidar/latent"] = latent
        data["diffusion/target_noise"] = noise
        return data

    @torch.no_grad()
    def forward_test(self,
                     data: Dict[str, Any],
                     use_gumbel: bool = False,
                     skip_act_and_threshold: bool = False,
                     num_steps: int = 50,
                     scheduler: SchedulerMixin = None,
                     threshold: float = 0.8,
                     gumbel_tau: float = 2.0,
                     generate_video: bool = False,
                     schedule_config: ScheduleConfig = ScheduleConfig(),
                     cfg_scale=4.0) -> Dict[str, Any]:
        batch_size = data["lidar"].shape[0]

        with torch.no_grad():
            cond = self.map_vae.encode(data["bev"]).latent_dist.sample()
        
        latents = self.sample_model(cond, cfg_scale, num_steps, device=data["lidar"].device, scheduler=self.scheduler if (scheduler is None) else scheduler, generate_list=generate_video, schedule_config=schedule_config)

        decoded = [self.vae.decode(l / self.vae.config.scaling_factor).sample for l in latents]

        current_dtype = decoded[0].dtype
        
        decoded_field = [d[:].detach().cpu() for d in decoded]
        
        decoded_field = [torch.nn.functional.sigmoid(d) for d in decoded_field]

        data["sample_field"] = decoded_field[-1]

        data["video_field"] = decoded_field

        return data


    def rescale_noise_cfg(self, noise_cfg, noise_pred_cond, guidance_rescale=0.7):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg


    def sample_model(self, cond, guidance_scale, sampling_steps,  device, scheduler, generate_list=False, schedule_config=ScheduleConfig()):

        generated_list = []

        batch_size = cond.shape[0]

        do_classifier_guidance = True
        cond_input = torch.cat([cond*0.0, cond]) if do_classifier_guidance else cond

        scheduler_timesteps_copy = scheduler.config.num_train_timesteps
        
        latent_size = 8

        latents = torch.randn((batch_size, latent_size,
                     40,
                     40)).to(device)


        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(sampling_steps)

        for i in range(scheduler.timesteps.shape[0] + schedule_config.second_to_last_noise_samples + schedule_config.last_noise_samples):

            t, skip_noise, append_now = self.schedule_cleanup_settings(i, scheduler, schedule_config)
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_guidance else latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_prediction = self.unet(latent_model_input, cond_input, t).sample

            if(do_classifier_guidance):
                
                noise_pred_uncond, noise_pred_cond = noise_prediction.chunk(2)
                noise_prediction = noise_pred_uncond  + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                #TODO: Scale? https://github.com/huggingface/diffusers/blob/58f5f748f4c8f63db787238392738ba3b377dbe4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L692C1-L692C115
                #noise_prediction = self.rescale_noise_cfg(noise_prediction, noise_pred_cond)
            
            if(self.scheduler is scheduler):
                latents = scheduler.step(noise_prediction, t, latents).prev_sample
            else:
                latents = scheduler.step(noise_prediction, t, latents, skip_noise=skip_noise).prev_sample

            if(generate_list and append_now):
                generated_list.append(latents)
        
        scheduler.set_timesteps(scheduler_timesteps_copy)

        if(generate_list):
            return generated_list

        return [latents]
        

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.training:
            return self.forward_train(data)
        else:
            return self.forward_test(data)



# import diffusers

# diffusers.models.unet_2d_condition.UNet2DConditionModel
#
# <class 'transformers.models.clip.tokenization_clip.CLIPTokenizer'>
# <class 'transformers.models.clip.modeling_clip.CLIPTextModel'>
# <class 'diffusers.schedulers.scheduling_pndm.PNDMScheduler'>
# <class 'diffusers.models.unet_2d_condition.UNet2DConditionModel'>
