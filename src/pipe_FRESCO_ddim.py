from src.utils import *
from src.flow_utils import warp_tensor
from typing import Optional
import torch
import torchvision
import gc
from diffusers.utils import torch_utils
import glob
from src.tokenflow_utils import register_batch_idx, register_pivotal

"""
DDIM Step

"""
def step_ddim(
        pipe,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        repeat_noise = False,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        visualize_pipeline=False, 
        flows=None, 
        occs=None,
        saliency=None
    ) :
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        scheduler = pipe.scheduler

        if scheduler.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf


        # if self.config.prediction_type == "epsilon":
        #     pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        #     pred_epsilon = model_output
        # elif self.config.prediction_type == "sample":
        #     pred_original_sample = model_output
        #     pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        # elif self.config.prediction_type == "v_prediction":
        #     pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        #     pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        # else:
        #     raise ValueError(
        #         f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
        #         " `v_prediction`"
        #     )

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        # BackGround smoothing in DDIM will cause background distortion. temporarily disabled.
        """
        [HACK] add background smoothing
        decode the feature
        warp the feature of f_{i-1}
        fuse the warped f_{i-1} with f_{i} in the non-salient region (i.e., background)
        encode the fused feature
        """
        # if saliency is not None and flows is not None and occs is not None:
        #     print('s')
        #     image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        #     image = warp_tensor(image, flows, occs, saliency, unet_chunk_size=1)
        #     pred_original_sample = pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.sample()  

        # 4. Clip or threshold "predicted x_0"

        # if self.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = scheduler._get_variance(timestep, prev_timestep)
        
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = torch_utils.randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )


            if repeat_noise:
                variance = variance[0:1].repeat(model_output.shape[0],1,1,1)
        
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance


        if visualize_pipeline: # for debug
            image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
            viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
            visualize(viz.cpu(), 90)

        if not return_dict:
            return (prev_sample,)

        return (prev_sample, pred_original_sample)


def ddim_step(scheduler, model_output, timestep, sample, generator, visualize_pipeline=False, flows=None, occs=None, saliency=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    
    if saliency is not None and flows is not None and occs is not None:
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        image = warp_image(image, flows, occs, saliency, unet_chunk_size=1)
        pred_original_sample = pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.sample()        
    
    pred_epsilon = model_output
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    #variance = scheduler._get_variance(timestep, prev_timestep)
    #std_dev_t = eta * variance ** (0.5)
    std_dev_t = 0
    
    if visualize_pipeline:
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
        visualize(viz.cpu(), 90)

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon     
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    
    return (prev_sample, pred_original_sample)
"""
==========================================================================
* step(): one DDPM step with background smoothing 
* inference(): translate one batch with FRESCO and background smoothing
==========================================================================
"""

def step(pipe, model_output, timestep, sample, generator, repeat_noise=False, 
         visualize_pipeline=False, flows=None, occs=None, saliency=None):
    """
    DDPM step with background smoothing
    * background smoothing: warp the background region of the previous frame to the current frame
    """
    scheduler = pipe.scheduler
    # 1. get previous step value (=t-1)
    prev_timestep = scheduler.previous_timestep(timestep)

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.one

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t    
    
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    """
    [HACK] add background smoothing
    decode the feature
    warp the feature of f_{i-1}
    fuse the warped f_{i-1} with f_{i} in the non-salient region (i.e., background)
    encode the fused feature
    """
    if saliency is not None and flows is not None and occs is not None:
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        image = warp_tensor(image, flows, occs, saliency, unet_chunk_size=1)
        pred_original_sample = pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.sample()    
    
    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t    
    
    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample    
    
    
    variance = beta_prod_t_prev / beta_prod_t * current_beta_t
    variance = torch.clamp(variance, min=1e-20)
    variance = (variance ** 0.5) * torch.randn(model_output.shape, generator=generator, 
                                               device=model_output.device, dtype=model_output.dtype)
    """
    [HACK] background smoothing
    applying the same noise could be good for static background
    """    
    if repeat_noise:
        variance = variance[0:1].repeat(model_output.shape[0],1,1,1)
        
    if visualize_pipeline: # for debug
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
        visualize(viz.cpu(), 90)

    pred_prev_sample = pred_prev_sample + variance
    
    return (pred_prev_sample, pred_original_sample)


@torch.no_grad()
def inference(pipe, controlnet, frescoProc, 
              imgs, prompt_embeds, edges, timesteps,
              cond_scale=[0.7]*20, num_inference_steps=20, num_warmup_steps=6, 
              do_classifier_free_guidance=True, seed=0, guidance_scale=7.5, use_controlnet=True,         
              record_latents=[], propagation_mode=False, visualize_pipeline=False, 
              flows = None, occs = None, saliency=None, repeat_noise=False,
              num_intraattn_steps = 1, step_interattn_end = 350, bg_smoothing_steps = [16,17], use_inv_noise = False, inv_noise = None, img_idx=None):
    """
    video-to-video translation inference pipeline with FRESCO
    * add controlnet and SDEdit
    * add FRESCO-guided attention
    * add FRESCO-guided optimization
    * add background smoothing
    * add support for inter-batch long video translation
    
    [input of the original pipe]
    pipe: base diffusion model
    imgs: a batch of the input frames
    prompt_embeds: prompts
    num_inference_steps: number of DDPM steps 
    timesteps: generated by pipe.scheduler.set_timesteps(num_inference_steps)
    do_classifier_free_guidance: cfg, should be always true
    guidance_scale: cfg scale
    seed

    [input of SDEdit]
    num_warmup_steps: skip the first num_warmup_steps DDPM steps

    [input of controlnet]
    use_controlnet: bool, whether using controlnet
    controlnet: controlnet model
    edges: input for controlnet (edge/stroke/depth, etc.)
    cond_scale: controlnet scale

    [input of FRESCO]
    frescoProc: FRESCO attention controller 
    flows: optical flows 
    occs: occlusion mask
    num_intraattn_steps: apply num_interattn_steps steps of spatial-guided attention
    step_interattn_end: apply temporal-guided attention in [step_interattn_end, 1000] steps

    [input for background smoothing]
    saliency: saliency mask
    repeat_noise: bool, use the same noise for all frames
    bg_smoothing_steps: apply background smoothing in bg_smoothing_steps

    [input for long video translation]
    record_latents: recorded latents in the last batch
    propagation_mode: bool, whether this is not the first batch
    
    [output]
    latents: a batch of latents of the translated frames 
    """
    gc.collect()
    torch.cuda.empty_cache()

    device = pipe._execution_device
    noise_scheduler = pipe.scheduler 
    generator = torch.Generator(device=device).manual_seed(seed)
    B, C, H, W = imgs.shape
    latents = pipe.prepare_latents(
        B,
        pipe.unet.config.in_channels,
        H,
        W,
        prompt_embeds.dtype,
        device,
        generator,
        latents = None,
    )
    if use_inv_noise:
        # print(img_idx)
        # s = [inv_noise[i] for i in img_idx]
        # print(len(s))
        latents = torch.cat([inv_noise[i].unsqueeze(0) for i in img_idx])
    # print('shape:',latents.shape)
    if repeat_noise:
        latents = latents[0:1].repeat(B,1,1,1).detach()
        
    if num_warmup_steps < 0:
        latents_init = latents.detach()
        num_warmup_steps = 0
    else:
        # SDEdit, use the noisy latent of imges as the input rather than a pure gausssian noise
        latent_x0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs.to(pipe.unet.dtype)).latent_dist.sample()
        latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()

    # SDEdit, run num_inference_steps-num_warmup_steps steps
    with pipe.progress_bar(total=num_inference_steps-num_warmup_steps) as progress_bar:
        latents = latents_init
        for i, t in enumerate(timesteps[num_warmup_steps:]):
            """
            [HACK] control the steps to apply spatial/temporal-guided attention
            [HACK] record and restore latents from previous batch
            """
            if i >= num_intraattn_steps:
                frescoProc.controller.disable_intraattn()
            if t < step_interattn_end:
                frescoProc.controller.disable_interattn()
            if propagation_mode: # restore latent from previous batch and record latent of the current batch
                latents[0:2] = record_latents[i].detach().clone()
                record_latents[i] = latents[[0,len(latents)-1]].detach().clone()
            else: # frist batch, record_latents[0][t] = [x_1,t, x_{N,t}] 
                record_latents += [latents[[0,len(latents)-1]].detach().clone()]
            
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            if use_controlnet:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

                down_block_res_samples, mid_block_res_sample = controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=edges,
                    conditioning_scale=cond_scale[i+num_warmup_steps],
                    guess_mode=False,
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None 
            
            # predict the noise residual
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            
            
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            """
            [HACK] background smoothing
            Note: bg_smoothing_steps should be rescaled based on num_inference_steps
            current [16,17] is based on num_inference_steps=20
            """
            if i + num_warmup_steps in bg_smoothing_steps:
                latents = step_ddim(pipe=pipe, 
                                    model_output=noise_pred, 
                                    timestep=t, 
                                    sample=latents, 
                                    generator=generator, 
                                    visualize_pipeline=visualize_pipeline, 
                                    flows = flows, 
                                    occs = occs, 
                                    saliency=saliency)[0]  
            else:
                latents = step_ddim(pipe=pipe, 
                                    model_output=noise_pred, 
                                    timestep=t, 
                                    sample=latents, 
                                    generator=generator, 
                                    visualize_pipeline=visualize_pipeline)[0]                            

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
                
    return latents

@torch.autocast(dtype=torch.float16, device_type='cuda')
@torch.no_grad()
def inference_pnp(pipe, frescoProc, 
              imgs, prompt_embeds, edges, timesteps,
              cond_scale=[0.7]*20, num_inference_steps=20, num_warmup_steps=6, 
              do_classifier_free_guidance=True, seed=0, guidance_scale=7.5,         
              record_latents=[], propagation_mode=False, visualize_pipeline=False, 
              flows = None, occs = None, saliency=None, repeat_noise=False,
              num_intraattn_steps = 1, step_interattn_end = 350, bg_smoothing_steps = [16,17], 
              inv_latent_path = None, img_idx=None, use_tokenflow = False):
    """
    video-to-video translation inference pipeline with FRESCO
    * add controlnet and SDEdit
    * add FRESCO-guided attention
    * add FRESCO-guided optimization
    * add background smoothing
    * add support for inter-batch long video translation
    
    [input of the original pipe]
    pipe: base diffusion model
    imgs: a batch of the input frames
    prompt_embeds: prompts
    num_inference_steps: number of DDPM steps 
    timesteps: generated by pipe.scheduler.set_timesteps(num_inference_steps)
    do_classifier_free_guidance: cfg, should be always true
    guidance_scale: cfg scale
    seed

    [input of SDEdit]
    num_warmup_steps: skip the first num_warmup_steps DDPM steps

    [input of controlnet]
    use_controlnet: bool, whether using controlnet
    controlnet: controlnet model
    edges: input for controlnet (edge/stroke/depth, etc.)
    cond_scale: controlnet scale

    [input of FRESCO]
    frescoProc: FRESCO attention controller 
    flows: optical flows 
    occs: occlusion mask
    num_intraattn_steps: apply num_interattn_steps steps of spatial-guided attention
    step_interattn_end: apply temporal-guided attention in [step_interattn_end, 1000] steps

    [input for background smoothing]
    saliency: saliency mask
    repeat_noise: bool, use the same noise for all frames
    bg_smoothing_steps: apply background smoothing in bg_smoothing_steps

    [input for long video translation]
    record_latents: recorded latents in the last batch
    propagation_mode: bool, whether this is not the first batch
    
    [output]
    latents: a batch of latents of the translated frames 
    """
    gc.collect()
    torch.cuda.empty_cache()

    device = pipe._execution_device
    noise_scheduler = pipe.scheduler 
    generator = torch.Generator(device=device).manual_seed(seed)
    B, C, H, W = imgs.shape
    # latents = pipe.prepare_latents(
    #     B,
    #     pipe.unet.config.in_channels,
    #     H,
    #     W,
    #     prompt_embeds.dtype,
    #     device,
    #     generator,
    #     latents = None,
    # )
    noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(inv_latent_path, f'noisy_latents_*.pt'))])
    latents_path = os.path.join(inv_latent_path, f'noisy_latents_{noisest}.pt')
    
    inv_noise = torch.load(latents_path)
    
    latents_init = torch.cat([inv_noise[i].unsqueeze(0) for i in img_idx])
    
    if repeat_noise:
        latents_init = latents[0:1].repeat(B,1,1,1).detach()
        
    if use_tokenflow:
        tokenflow_img_idx = [i for i in range(img_idx[0], img_idx[-1]+1)]
        if propagation_mode:
            tokenflow_img_idx = tokenflow_img_idx[0:1] + [i for i in range(img_idx[1], img_idx[-1]+1)]

        print(tokenflow_img_idx)
    
    
    # if num_warmup_steps < 0:
    #     latents_init = latents.detach()
    #     num_warmup_steps = 0
    # else:
    #     # SDEdit, use the noisy latent of imges as the input rather than a pure gausssian noise
    #     latent_x0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs.to(pipe.unet.dtype)).latent_dist.sample()
    #     latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()

    # Plug and play edit, run num_inference_steps
    if use_tokenflow:
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            latents_tokenflow = torch.cat([inv_noise[i].unsqueeze(0) for i in tokenflow_img_idx])
            latents = latents_init
            for i, t in enumerate(timesteps):
                """
                [HACK] control the steps to apply spatial/temporal-guided attention
                [HACK] record and restore latents from previous batch
                """
                if i >= num_intraattn_steps:
                    frescoProc.controller.disable_intraattn()
                if t < step_interattn_end:
                    frescoProc.controller.disable_interattn()
                if propagation_mode: # restore latent from previous batch and record latent of the current batch
                    latents[0:2] = record_latents[i].detach().clone()
                    record_latents[i] = latents[[0,len(latents)-1]].detach().clone()
                else: # frist batch, record_latents[0][t] = [x_1,t, x_{N,t}] 
                    record_latents += [latents[[0,len(latents)-1]].detach().clone()]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                ref_noise = load_source_latents_t(t,inv_latent_path)

                ref_latent = torch.cat([ref_noise[i].unsqueeze(0) for i in img_idx])
                # print(img_idx)
                latent_model_input = torch.cat([ref_latent,latent_model_input])
                print('keyframe_shape:',latent_model_input.shape)


                down_block_res_samples, mid_block_res_sample = None, None 

                
                register_pivotal(pipe.unet, True)
                register_time(pipe, t)
                # self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
                
                
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                register_pivotal(pipe.unet, False)

                
            
                denoised_latents = []
                # for i, b in enumerate(range(0, len(tokenflow_img_idx), tokenflow_batch_size)):
                for i, b in enumerate(img_idx[:-1]):
                    

                    end = img_idx[i+1]+1 if b == img_idx[-2] else img_idx[i+1]

                    if propagation_mode:
                        if b == 0:
                            end = 1
                        else:
                            b = b - img_idx[1]+1
                            end = end - img_idx[1]+1

                    
                    
                    latent_model_input_tokenflow = torch.cat([latents_tokenflow[b:end]] * 2) if do_classifier_free_guidance else latents_tokenflow[b:end]
                    ref_latent_tokenflow = torch.cat([ref_noise[i].unsqueeze(0) for i in tokenflow_img_idx[b:end]])
                    latent_model_input_tokenflow = torch.cat([ref_latent_tokenflow, latent_model_input_tokenflow])
                    print('batch_num:',i,'batch:',b)
                    register_batch_idx(pipe.unet, i)

                    # print('tokenflow_shape:',latent_model_input_tokenflow.shape)
                    pnp_prompt_tensorflow = torch.cat([prompt_embeds.chunk(3)[0][0].unsqueeze(0)] * (end - b))
                    nprompt_tensorflow = torch.cat([prompt_embeds.chunk(3)[1][0].unsqueeze(0)] * (end - b))
                    prompt_tensorflow = torch.cat([prompt_embeds.chunk(3)[2][0].unsqueeze(0)] * (end - b))
                    prompt_embeds_tokenflow = torch.cat([pnp_prompt_tensorflow, nprompt_tensorflow, prompt_tensorflow])
                    # print(pnp_prompt_tensorflow.shape, prompt_tensorflow.shape, nprompt_tensorflow.shape)
                    # print('new prompt',prompt_embeds_tokenflow.shape)

                    noise_pred = pipe.unet(
                        latent_model_input_tokenflow,
                        t,
                        encoder_hidden_states=prompt_embeds_tokenflow,
                        cross_attention_kwargs=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                    if do_classifier_free_guidance:
                        _,noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    """
                    [HACK] background smoothing
                    Note: bg_smoothing_steps should be rescaled based on num_inference_steps
                    current [16,17] is based on num_inference_steps=20
                    """
                    if i + num_warmup_steps in bg_smoothing_steps:
                        latents_batch = step_ddim(pipe=pipe, 
                                            model_output=noise_pred, 
                                            timestep=t, 
                                            sample=latents_tokenflow[b:end], 
                                            generator=generator, 
                                            visualize_pipeline=visualize_pipeline, 
                                            flows = flows, 
                                            occs = occs, 
                                            saliency=saliency)[0]  
                    else:
                        latents_batch = step_ddim(pipe=pipe, 
                                            model_output=noise_pred, 
                                            timestep=t, 
                                            sample=latents_tokenflow[b:end], 
                                            generator=generator, 
                                            visualize_pipeline=visualize_pipeline)[0] 
                    denoised_latents.append(latents_batch)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                        # denoised_latents.append(pipe.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
                latents_tokenflow = torch.cat(denoised_latents)
                if propagation_mode:
                    img_idx_propagate = [i-img_idx[1]+1 for i in img_idx]
                    img_idx_propagate[0] = 0
                    latents = torch.cat([latents_tokenflow[i].unsqueeze(0) for i in img_idx_propagate])
                    # print(img_idx_propagate)
                else:
                    latents = torch.cat([latents_tokenflow[i].unsqueeze(0) for i in img_idx])
        
        return latents_tokenflow


    # Plug and play edit, run num_inference_steps
    else:        
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            latents = latents_init
            for i, t in enumerate(timesteps):
                """
                [HACK] control the steps to apply spatial/temporal-guided attention
                [HACK] record and restore latents from previous batch
                """
                print(frescoProc.controller.use_intraattn)
                if i >= num_intraattn_steps:
                    frescoProc.controller.disable_intraattn()
                if t < step_interattn_end:
                    frescoProc.controller.disable_interattn()
                if propagation_mode: # restore latent from previous batch and record latent of the current batch
                    latents[0:2] = record_latents[i].detach().clone()
                    record_latents[i] = latents[[0,len(latents)-1]].detach().clone()
                else: # frist batch, record_latents[0][t] = [x_1,t, x_{N,t}] 
                    record_latents += [latents[[0,len(latents)-1]].detach().clone()]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                ref_noise = load_source_latents_t(t,inv_latent_path)

                ref_latent = torch.cat([ref_noise[i].unsqueeze(0) for i in img_idx])
                # print(img_idx)
                latent_model_input = torch.cat([ref_latent,latent_model_input])


                down_block_res_samples, mid_block_res_sample = None, None 

                register_time(pipe, t)

                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]


                # perform guidance (in pnp mode)
                if do_classifier_free_guidance:
                    _,noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                """
                [HACK] background smoothing
                Note: bg_smoothing_steps should be rescaled based on num_inference_steps
                current [16,17] is based on num_inference_steps=20
                """
                if i + num_warmup_steps in bg_smoothing_steps:
                    latents = step_ddim(pipe=pipe, 
                                        model_output=noise_pred, 
                                        timestep=t, 
                                        sample=latents, 
                                        generator=generator, 
                                        visualize_pipeline=visualize_pipeline, 
                                        flows = flows, 
                                        occs = occs, 
                                        saliency=saliency)[0]  
                else:
                    latents = step_ddim(pipe=pipe, 
                                        model_output=noise_pred, 
                                        timestep=t, 
                                        sample=latents, 
                                        generator=generator, 
                                        visualize_pipeline=visualize_pipeline)[0]                            

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                
        return latents
