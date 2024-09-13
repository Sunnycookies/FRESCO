from src.utils import *
from src.flow_utils import warp_tensor
import torch
import torchvision
import gc
from diffusers.utils import torch_utils

from src.diffusion_hacked import *
from src.tokenflow_utils import register_batch_idx, register_pivotal
from src.noise_warp import *
from src.tokenflow_utils import *
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

"""
DDIM Step

"""
def step_ddim(pipe, model_output, timestep, sample, eta = 0.0, use_clipped_model_output = False, generator = None, repeat_noise = False,
              variance_noise = None, return_dict = True, visualize_pipeline = False, flows = None, occs = None,saliency = None) :
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
                variance_noise = torch_utils.randn_tensor(model_output.shape, generator=generator, 
                                                          device=model_output.device, dtype=model_output.dtype
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

def step_warp(pipe, model_output, timestep, sample, generator, repeat_noise=False, 
         visualize_pipeline=False, flows=None, occs=None, saliency=None, warp_noise = False, flows_centralized = None):
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
    
    
    vari = beta_prod_t_prev / beta_prod_t * current_beta_t
    vari = torch.clamp(vari, min=1e-20)
    variance = (vari ** 0.5) * torch.randn(model_output.shape, generator=generator, 
                                               device=model_output.device, dtype=model_output.dtype)
    """
    [HACK] background smoothing
    applying the same noise could be good for static background
    """    
    if repeat_noise:
        variance = variance[0:1].repeat(model_output.shape[0],1,1,1)

    if warp_noise:
	    latent_type = variance.dtype
	    var_multiplier = torch.randn(model_output.shape, generator=generator, 
            device=model_output.device, dtype=model_output.dtype)
	    var_multiplier = warp_pure_noise(var_multiplier, flows_centralized, occs, saliency, 1).to(latent_type)
	    variance = (vari ** 0.5) * var_multiplier

        
    if visualize_pipeline: # for debug
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
        visualize(viz.cpu(), 90)

    pred_prev_sample = pred_prev_sample + variance
    
    return (pred_prev_sample, pred_original_sample)

@torch.autocast(dtype=torch.float16, device_type='cuda')
@torch.no_grad()
def inference(pipe, controlnet, frescoProc, 
              imgs, prompt_embeds, edges, timesteps,
              cond_scale=[0.7]*20, num_inference_steps=20, num_warmup_steps=6, 
              do_classifier_free_guidance=True, seed=0, guidance_scale=7.5, use_controlnet=True,         
              record_latents=[], propagation_mode=False, visualize_pipeline=False, 
              flows = None, occs = None, saliency=None, repeat_noise=False,
              num_intraattn_steps = 1, step_interattn_end = 350, bg_smoothing_steps = [16,17], img_idx = None, 
              use_tokenflow = False, imgs_all = None, edges_all = None, warp_noise = False, 
              flows_centralized = None, flow_model = None, sod_model = None, dilate = None):
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

    if warp_noise:
        latents_type = latents.dtype
        latents = warp_pure_noise(latents, flows_centralized, occs, saliency, 1)
        latents = latents.to(latents_type)
   
    
    if repeat_noise:
        latents = latents[0:1].repeat(B,1,1,1).detach()
        
    if num_warmup_steps < 0:
        latents_init = latents.detach()
        num_warmup_steps = 0
    else:
        # SDEdit, use the noisy latent of imges as the input rather than a pure gausssian noise
        latent_x0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs.to(pipe.unet.dtype)).latent_dist.sample()
        latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()

    if use_tokenflow:
        imgs_all_torch = torch.cat([numpy2tensor(i) for i in imgs_all], dim=0)
        B, C, H, W = imgs_all_torch.shape
        latents_all = pipe.prepare_latents(
            B,
            pipe.unet.config.in_channels,
            H,
            W,
            prompt_embeds.dtype,
            device,
            generator,
            latents = None,
        )

        if repeat_noise:
            latents_all = latents[0:1].repeat(B,1,1,1).detach()

        
        if num_warmup_steps < 0:
            latents_init_all = latents.detach()
            num_warmup_steps = 0
        else:
            # SDEdit, use the noisy latent of imges as the input rather than a pure gausssian noise
            batch_size = 8
            latent_x0_all = []
            fwd_occs_all = []
            bwd_occs_all = []
            fwd_flows_centralized_all = []
            bwd_flows_centralized_all = []
            fwd_flows_all = []
            bwd_flows_all = []
            saliency_all = []

            for i in range(0,len(imgs_all),batch_size):
                end = i + batch_size if i + batch_size < len(imgs_all) else len(imgs_all)
                latents_batch = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs_all_torch[i:end].to(pipe.unet.dtype)).latent_dist.sample()
                # latent_x0_all.append(pipe.vae.config.scaling_factor * pipe.vae.encode(imgs_all[i:end].to(pipe.unet.dtype)).latent_dist.sample())

                latents_type = latents.dtype
                flows_batch, occs_batch, _, _, flows_centralized_batch = get_flow_and_interframe_paras_warped(flow_model, imgs_all[ i:end ])
                saliency_batch = get_saliency(imgs_all[i:end], sod_model, dilate)
                print(flows_batch[0].shape, occs_batch[0].shape, flows_centralized_batch[0].shape,saliency_batch.shape)
                fwd_flows_all.append(flows_batch[0])
                bwd_flows_all.append(flows_batch[1])
                fwd_occs_all.append(occs_batch[0])
                bwd_occs_all.append(occs_batch[1])
                fwd_flows_centralized_all.append(flows_centralized_batch[0])
                bwd_flows_centralized_all.append(flows_centralized_batch[1])
                saliency_all.append(saliency_batch)

                if warp_noise:
                    # print(saliency_all.shape, occs_all[1].shape, flows_centralized_all[1].shape)
                    latents_batch = warp_pure_noise(latents_batch, flows_centralized_batch, occs_batch, saliency_batch, 1)
                    latents_batch = latents_batch.to(latents_type)
                
                latent_x0_all.append(latents_batch)


            latent_x0_all = torch.cat(latent_x0_all)
            fwd_occs_all = torch.cat(fwd_occs_all)
            bwd_occs_all = torch.cat(bwd_occs_all)
            fwd_flows_centralized_all = torch.cat(fwd_flows_centralized_all)
            bwd_flows_centralized_all = torch.cat(bwd_flows_centralized_all)
            fwd_flows_all = torch.cat(fwd_flows_all)
            bwd_flows_all = torch.cat(bwd_flows_all)
            saliency_all = torch.cat(saliency_all)
            # latent_x0_all = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs_all.to(pipe.unet.dtype)).latent_dist.sample()
            latents_init_all = noise_scheduler.add_noise(latent_x0_all, latents_all, timesteps[num_warmup_steps]).detach()  


        tokenflow_img_idx = [i for i in range(img_idx[0], img_idx[-1]+1)]
        if propagation_mode:
            tokenflow_img_idx = tokenflow_img_idx[0:1] + [i for i in range(img_idx[1], img_idx[-1]+1)]

        print(tokenflow_img_idx)
    # SDEdit, run num_inference_steps-num_warmup_steps steps
    if use_tokenflow:
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            latents_tokenflow = torch.cat([latents_init_all[i].unsqueeze(0) for i in tokenflow_img_idx])
            edges_tokenflow = torch.cat([edges_all[i].unsqueeze(0) for i in tokenflow_img_idx])
            imgs_tokenflow = [imgs_all[i] for i in tokenflow_img_idx]
            fwd_flows_tokenflow = torch.cat([fwd_flows_all[i].unsqueeze(0) for i in tokenflow_img_idx])
            bwd_flows_tokenflow = torch.cat([bwd_flows_all[i].unsqueeze(0) for i in tokenflow_img_idx])
            fwd_occs_tokenflow = torch.cat([fwd_occs_all[i].unsqueeze(0) for i in tokenflow_img_idx ])
            bwd_occs_tokenflow = torch.cat([bwd_occs_all[i].unsqueeze(0) for i in tokenflow_img_idx ])
            fwd_flows_centralized_tokenflow = torch.cat([fwd_flows_centralized_all[i].unsqueeze(0) for i in tokenflow_img_idx ])
            bwd_flows_centralized_tokenflow = torch.cat([bwd_flows_centralized_all[i].unsqueeze(0) for i in tokenflow_img_idx ])
            saliency_tokenflow = torch.cat([saliency_all[i].unsqueeze(0) for i in tokenflow_img_idx ])
            print(fwd_flows_tokenflow.shape, bwd_flows_tokenflow.shape, fwd_occs_tokenflow.shape, bwd_occs_tokenflow.shape, fwd_flows_centralized_tokenflow.shape, bwd_flows_centralized_tokenflow.shape)
            # print(edges_tokenflow.shape)
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

                register_pivotal(pipe.unet, True)
                register_time(pipe, t)

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
                for i, b in enumerate(img_idx[:-1]):
                    end = img_idx[i+1]+1 if b == img_idx[-2] else img_idx[i+1]

                    if propagation_mode:
                        if b == 0:
                            end = 1
                        else:
                            b = b - img_idx[1]+1
                            end = end - img_idx[1]+1
                    
                    latent_model_input_tokenflow = torch.cat([latents_tokenflow[b:end]] * 2) if do_classifier_free_guidance else latents_tokenflow[b:end]
                    # ref_latent_tokenflow = torch.cat([ref_noise[i].unsqueeze(0) for i in tokenflow_img_idx[b:end]])
                    # latent_model_input_tokenflow = torch.cat([ref_latent_tokenflow, latent_model_input_tokenflow])
                    print('batch_num:',i,'batch:',b)
                    register_batch_idx(pipe.unet, i) 
                    nprompt_tensorflow = torch.cat([prompt_embeds.chunk(3)[1][0].unsqueeze(0)] * (end - b))
                    prompt_tensorflow = torch.cat([prompt_embeds.chunk(3)[2][0].unsqueeze(0)] * (end - b))
                    prompt_embeds_tokenflow = torch.cat([nprompt_tensorflow, prompt_tensorflow])

                    if use_controlnet:
                        control_model_input = latent_model_input_tokenflow
                        controlnet_prompt_embeds = prompt_embeds_tokenflow
                        edges_tokenflow_input = torch.cat([edges_tokenflow[b:end]] * 2)
                        # print(edges_tokenflow_input.shape)

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=edges_tokenflow_input,
                            conditioning_scale=cond_scale[i+num_warmup_steps],
                            guess_mode=False,
                            return_dict=False,
                        )
                    else:
                        down_block_res_samples, mid_block_res_sample = None, None

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
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    flows_, occs_, saliency_ = None, None, None
                    if warp_noise:
                        flows_centralized_ = [fwd_flows_centralized_tokenflow[b:end], bwd_flows_centralized_tokenflow[b:end]]
                        if i + num_warmup_steps in bg_smoothing_steps:
                            flows_ = [torch.cat(fwd_flows_tokenflow[b:end]), torch.cat(bwd_flows_tokenflow[b:end])]
                            occs_ = [torch.cat(fwd_occs_tokenflow[b:end]), torch.cat(bwd_occs_tokenflow[b:end])]
                            flows_centralized_ = [torch.cat(flows) for flows in flows_centralized_]
                        latents_batch = step_warp(pipe, noise_pred, t, latents_tokenflow[b:end], generator, visualize_pipeline=visualize_pipeline,
                                                  flows=flows_, occs=occs_, saliency=saliency_, flows_centralized=flows_centralized_, warp_noise=True)[0]

                    else:
                        if i + num_warmup_steps in bg_smoothing_steps:
                            flows_, occs_, saliency_ = flows, occs, saliency
                        latents_batch = step(pipe, noise_pred, t, latents_tokenflow[b:end], generator, flows=flows_, occs=occs_, 
                                             visualize_pipeline=visualize_pipeline, saliency=saliency_)[0]
                        # latents_batch = step_ddim(
                        #     pipe = pipe,
                        #     model_output=noise_pred, 
                        #     timestep=t, 
                        #     sample=latents_tokenflow[b:end], 
                        #     generator=generator, 
                        #     visualize_pipeline=visualize_pipeline
                        # )[0]
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
                    print(img_idx_propagate)
                else:
                    latents = torch.cat([latents_tokenflow[i].unsqueeze(0) for i in img_idx])

        return latents_tokenflow.to(pipe.unet.dtype)
    else:
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
                flows_, occs_, saliency_ = None, None, None
                if i + num_warmup_steps in bg_smoothing_steps:
                        flows_, occs_, saliency_ = flows, occs, saliency
                if warp_noise:
                    print('warped')
                    latents = step_warp(pipe, noise_pred, t, latents, generator, 
                                        visualize_pipeline=visualize_pipeline, 
                                        flows = flows_, occs = occs_, saliency=saliency_, 
                                        flows_centralized = flows_centralized, warp_noise = True)[0]                          
                else:
                    latents = step(pipe, noise_pred, t, latents, generator, 
                                   visualize_pipeline=visualize_pipeline, 
                                   flows = flows_, occs = occs_, saliency=saliency_)[0]                        

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                
        return latents

@torch.autocast(dtype=torch.float16, device_type='cuda')
@torch.no_grad()
def inference_extended(pipe, controlnet, frescoProc, imgs, edges, timesteps, keylists, n_prompt, prompts, inv_prompts, 
                       end_opt_step, batch_size, propagation_mode, repeat_noise = False, warp_noise = False, 
                       do_classifier_free_guidance = True, use_tokenflow = False, edit_mode = 'SDEdit', visualize_pipeline = False, 
                       use_controlnet = True, use_saliency = False, use_inv_noise = False, cond_scale = [0.7] * 20, 
                       num_inference_steps = 20, num_warmup_steps = 6, seed = 0, guidance_scale = 7.5, record_latents = [], 
                       inv_noise = None, num_intraattn_steps = 1, step_interattn_end = 350, bg_smoothing_steps = [16, 17], 
                       flow_model = None, sod_model = None, dilate = None):

    gc.collect()
    torch.cuda.empty_cache()

    if not use_tokenflow and len(keylists) > 1:
        print("Ebsynth method requires keyframes to be fixed!")
        raise NotImplementedError

    device = pipe._execution_device
    noise_scheduler = pipe.scheduler 
    generator = torch.Generator(device=device).manual_seed(seed)

    imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)

    # calculate for prompt_embed.dtype
    prompt_embeds = pipe._encode_prompt(
        prompts,
        device,
        1,
        do_classifier_free_guidance,
        [n_prompt] * len(prompts)
    )
    
    # prepate initial latents (noise)
    if edit_mode == 'pnp' or use_inv_noise:
        latents = inv_noise
    else:
        B, C, H, W = imgs_torch.shape
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

    if warp_noise:
        flows_all, occs_all, _, _, flows_centralized_all = get_flow_and_interframe_paras_warped(flow_model, imgs)
    else:
        flows_all, occs_all, _, _ = get_flow_and_interframe_paras(flow_model, imgs)

    if warp_noise:
        latents_type = latents.dtype
        saliency_all = get_saliency(imgs, sod_model, dilate) if use_saliency else None
        latents = warp_pure_noise(latents, flows_centralized_all, occs_all, saliency_all, 1)
        latents = latents.to(latents_type)
    
    if repeat_noise:
        latents = latents[0:1].repeat(B,1,1,1).detach()

    if num_warmup_steps < 0:
        latents_init = latents.detach()
        num_warmup_steps = 0
    else:
        latent_x0 = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs_torch.to(pipe.unet.dtype)).latent_dist.sample()
        latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()

    keys_loop_len = len(keylists)

    # memory used to store parameters
    flows_mem = []
    occs_mem = []
    flows_centralized_mem = []
    saliency_mem = []
    attn_mask_mem = []
    interattn_paras_mem = []
    correlation_matrix_mem = []

    # prepare parameters for inter-frame and intra-frame consistency
    frescoProc.controller.enable_intraattn(True)
    if use_tokenflow:
        deactivate_tokenflow(pipe.unet)

    for ind, keygroup in enumerate(keylists):
        imgs_group = [imgs[i] for i in keygroup]
        imgs_group_torch = imgs_torch[keygroup]
        prompt_embeds_group = pipe._encode_prompt(
            [prompts[i] for i in keygroup],
            device,
            1,
            do_classifier_free_guidance,
            [n_prompt] * len(keygroup)
        )
        if edit_mode == 'pnp':
            pnp_prompt_embeds_group = pipe._encode_prompt(
                [inv_prompts[i] for i in keygroup],
                device,
                1,
                False,
                ''
            )
            prompt_embeds_group = torch.cat([pnp_prompt_embeds_group, prompt_embeds_group])

        saliency_group = get_saliency(imgs_group, sod_model, dilate) if use_saliency else None

        if warp_noise:
            flows_group, occs_group, attn_mask_group, interattn_paras_group, \
                flows_centralized_group = get_flow_and_interframe_paras_warped(flow_model, imgs_group)
        else:
            flows_group, occs_group, attn_mask_group, interattn_paras_group \
                = get_flow_and_interframe_paras(flow_model, imgs_group)
            flows_centralized_group = None
        
        correlation_matrix_group = get_intraframe_paras(pipe, imgs_group_torch, frescoProc, prompt_embeds_group,
                                                        do_classifier_free_guidance, seed, False, True, ind)
        
        flows_mem.append(flows_group)
        occs_mem.append(occs_group)
        flows_centralized_mem.append(flows_centralized_group)
        saliency_mem.append(saliency_group)
        attn_mask_mem.append(attn_mask_group)
        interattn_paras_mem.append(interattn_paras_group)
        correlation_matrix_mem.append(correlation_matrix_group)
    
    if use_tokenflow:
        set_tokenflow(pipe.unet, edit_mode)

    gc.collect()
    torch.cuda.empty_cache()

    latents = latents_init

    with pipe.progress_bar(total=num_inference_steps - num_warmup_steps) as progress_bar:
        for i, t in enumerate(timesteps[num_warmup_steps:]):
            i_ = i % keys_loop_len
            keygroup = keylists[i_]

            # prepare a group of frame based on keygroup
            # print(f"processing keygroup [{i_ + 1}/{len(keylists)}] with keyframes {keygroup}")

            flows = flows_mem[i_]
            occs = occs_mem[i_]
            attn_mask = attn_mask_mem[i_]
            interattn_paras = interattn_paras_mem[i_]
            flows_centralized = flows_centralized_mem[i_]
            correlation_matrix = correlation_matrix_mem[i_]
            saliency = saliency_mem[i_]

            # Turn on all FRESCO support
            frescoProc.controller.enable_controller(interattn_paras, attn_mask, True, False, i_)
            apply_FRESCO_opt(pipe, steps = timesteps[:end_opt_step], flows = flows, occs = occs, 
                             correlation_matrix=correlation_matrix, saliency=saliency, optimize_temporal = True)

            if i >= num_intraattn_steps:
                frescoProc.controller.disable_intraattn(False)
            if t < step_interattn_end:
                frescoProc.controller.disable_interattn()
            if propagation_mode:
                latents[0:2] = record_latents[i].detach().clone()
                record_latents[i] = latents[[0,len(latents)-1]].detach().clone()
            else:
                record_latents += [latents[[0,len(latents)-1]].detach().clone()]

            latent_model_input = latents[keygroup]
            keyframe_edges = edges[keygroup]
            keyframe_prompt_embeds = pipe._encode_prompt(
                [prompts[k] for k in keygroup],
                device,
                1,
                do_classifier_free_guidance,
                [n_prompt] * len(keygroup)
            )
            if edit_mode == 'pnp':
                pnp_keyframe_prompt_embeds = pipe._encode_prompt(
                    [inv_prompts[k] for k in keygroup],
                    device,
                    1,
                    False,
                    ''
                )
                keyframe_prompt_embeds = torch.cat([pnp_keyframe_prompt_embeds, keyframe_prompt_embeds])

            if do_classifier_free_guidance:
                keyframe_edges = torch.cat([keyframe_edges.to(pipe.unet.dtype)] * 2)
                latent_model_input = torch.cat([latent_model_input] * 2)

            if use_controlnet:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = keyframe_prompt_embeds
                controlnet_cond = keyframe_edges

                down_block_res_samples, mid_block_res_sample = controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=cond_scale[i+num_warmup_steps],
                    guess_mode=False,
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None
            
            if use_tokenflow:
                register_pivotal(pipe.unet, True)
                register_time(pipe, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=keyframe_prompt_embeds,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            
            if use_tokenflow:
                register_pivotal(pipe.unet, False)
            elif do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            disable_FRESCO_opt(pipe)
            frescoProc.controller.disable_controller(False)

            if not use_tokenflow:
                flows_, occs_, saliency_ = None, None, None
                if i + num_warmup_steps in bg_smoothing_steps:
                    flows_, occs_, saliency_ = flows, occs, saliency
                if warp_noise:
                    flows_centralized_ = flows_centralized
                    latents = step_warp(pipe, noise_pred, t, latents, generator,
                                        visualize_pipeline=visualize_pipeline,
                                        flows=flows_, occs=occs_, saliency=saliency_,
                                        flows_centralized=flows_centralized_, warp_noise=True)[0]
                else:
                    latents = step(pipe, noise_pred, t, latents, generator,
                                   visualize_pipeline=visualize_pipeline,
                                   flows=flows_, occs=occs_, saliency=saliency_)[0]
                    
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                
                continue

            denoised_latents = []

            for j, key in enumerate(keygroup):
                end = len(latents) if key == keygroup[-1] else keygroup[j + 1]
                if propagation_mode and key == 0:
                    end = 1

                # print(f"conducting tokenflow on images [{key}:{end}]")

                register_batch_ind = j + (j > 0) - (key == keygroup[-1])

                register_batch_idx(pipe.unet, register_batch_ind)
                
                full_latent = latents[key:end]
                latent_model_input_tokenflow = full_latent
                edges_tokenflow = edges[key:end]
                prompt_embeds_tokenflow = pipe._encode_prompt(
                    prompts[key:end],
                    device,
                    1,
                    do_classifier_free_guidance,
                    [n_prompt] * (end - key)
                )
                if edit_mode == 'pnp':
                    pnp_prompt_embeds_tokenflow = pipe._encode_prompt(
                        inv_prompts[key:end],
                        device,
                        1,
                        False,
                        ''
                    )
                    prompt_embeds_tokenflow = torch.cat([pnp_prompt_embeds_tokenflow, prompt_embeds_tokenflow])

                if do_classifier_free_guidance:
                    edges_tokenflow = torch.cat([edges_tokenflow.to(pipe.unet.dtype)] * 2)
                    latent_model_input_tokenflow = torch.cat([full_latent] * 2)
                
                if use_controlnet:
                    control_model_input = latent_model_input_tokenflow
                    controlnet_prompt_embeds = prompt_embeds_tokenflow

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=edges_tokenflow,
                        conditioning_scale=cond_scale[i+num_warmup_steps],
                        guess_mode=False,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

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
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                flows_, occs_, saliency_ = None, None, None
                if i + num_warmup_steps in bg_smoothing_steps and end - key > 1:
                    flows_ = [flow[key:end] for flow in flows_all]
                    occs_ = [occ[key:end] for occ in occs_all]
                    if use_saliency:
                        # saliency_ = saliency[key:end]
                        saliency_ = get_saliency(imgs[key:end], sod_model, dilate)
                if warp_noise:
                    flows_centralized_ = [flow_c[key:end] for flow_c in flows_centralized_all]
                    latents_batch = step_warp(pipe, noise_pred, t, full_latent, generator, visualize_pipeline=visualize_pipeline,
                                              flows=flows_, occs=occs_, saliency=saliency_, flows_centralized=flows_centralized_,
                                              warp_noise=True)[0]
                else:
                    latents_batch = step(pipe, noise_pred, t, full_latent, generator, flows=flows_, occs=occs_, 
                                         visualize_pipeline=visualize_pipeline, saliency=saliency_)[0]
                
                denoised_latents.append(latents_batch)

            latents = torch.cat(denoised_latents)
        
            if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
        
    frescoProc.controller.clear_store()
    frescoProc.controller.disable_controller()

    gc.collect()
    torch.cuda.empty_cache()

    return latents         