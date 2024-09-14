import os
import yaml
import json
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

save_path = '/mnt/netdisk/linjunxin/fresco/'
user_path = '/home/linjx'
fresco_path = user_path + '/fresco'
tokenflow_path = user_path + '/fresco'

# edit_method = 'pnp'
edit_method = 'SDEdit'
# synth_method = 'Tokenflow'
# synth_method = 'ebsynth'
synth_method = 'None'

if edit_method == 'SDEdit':
    video_dir = fresco_path + '/data/videos'
else:
    video_dir = tokenflow_path + '/data/tokenflow_supp_videos'
prompts = video_dir + '/prompts.json'
key_intervs = video_dir + '/key_interv.json'
inv_prompts = video_dir + '/inv_prompts.json'

ref_yaml = fresco_path + '/config/ref_config.yaml'
refs = fresco_path + '/cfg.json'

run_ebsynth = False
use_warp_noise = False
use_saliency = True
use_inv_prompts = False
use_inv_noise = False
if edit_method == 'pnp':
    use_inv_noise = True
    use_inv_prompts = True

all_as_key = False or synth_method not in ['Tokenflow', 'ebsynth']
keyframe_select_mode = 'fixed'
keyframe_select_radix = 6
if all_as_key:
    keyframe_select_radix = 1

default_sd = 'stabilityai/stable-diffusion-2-1-base'

with open(prompts, 'r') as f:
    prompt_dict = json.load(f)
with open(key_intervs, 'r') as f:
    key_dict = json.load(f)
if use_inv_prompts == True:
    with open(inv_prompts, 'r') as f:
        inv_pronpts_dict = json.load(f)
with open(refs, 'r') as f:
    cfg_dict = json.load(f)

if not os.path.exists(os.path.join(fresco_path, 'model')):
    os.system(f"conda activate fresco")
    os.system(f"python {fresco_path}/install_mirror.py")

for name, prompts in prompt_dict.items():
    file_name, ext = os.path.splitext(name)

    # for debug
    print(prompts)

    # process item control
    if file_name not in ['tiger_input']:
        continue

    with open(ref_yaml,'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    config_yaml['file_path'] = os.path.join(video_dir, name)

    # methods selection
    config_yaml['edit_mode'] = edit_method
    config_yaml['synth_mode'] = synth_method
    config_yaml['warp_noise'] = use_warp_noise
    config_yaml['use_saliency'] = use_saliency
    config_yaml['run_ebsynth'] = run_ebsynth
    config_yaml['use_controlnet'] = edit_method == 'SDEdit'
    config_yaml['use_inv_noise'] = use_inv_noise
    
    # parameters for keyframe selection
    config_yaml['mininterv'] = key_dict[name][0]
    config_yaml['maxinterv'] = key_dict[name][1]
    if all_as_key:
        config_yaml['mininterv'] = config_yaml['maxinterv'] = 1
    config_yaml['keyframe_select_mode'] = keyframe_select_mode
    config_yaml['keyframe_select_radix'] = keyframe_select_radix

    # save video path
    suffix = f"test-{config_yaml['synth_mode']}-{config_yaml['edit_mode']}-"
    suffix += f"{config_yaml['keyframe_select_mode']}{'-warp' if use_warp_noise else ''}"
    save_path_video = os.path.join(save_path, suffix, file_name)
    os.makedirs(save_path_video, exist_ok=True)
    
    # model selection
    config_yaml['sd_path'] = default_sd
    
    # process parameters
    config_yaml['seed'] = 0
    config_yaml['batch_size'] = 4
    config_yaml['cond_scale'] = 0.7
    config_yaml['controlnet_type'] = 'hed'
    config_yaml['num_inference_steps'] = 20
    config_yaml['num_warmup_steps'] = 0
    config_yaml['end_opt_step'] = 40
    config_yaml['max_process'] = 4
    config_yaml['use_freeu'] = False
    config_yaml['inv_inference_steps'] = 500
    config_yaml['inv_batch_size'] = 20
    config_yaml['pnp_attn_t'] = 0.5
    config_yaml['pnp_f_t'] = 0.8

    # load customized parameters
    if file_name in cfg_dict:
        config_yaml['cond_scale'] = cfg_dict[file_name]['control_scales']
        config_yaml['controlnet_type'] = cfg_dict[file_name]['control']
        config_yaml['num_warmup_steps'] = int(config_yaml['num_inference_steps'] * cfg_dict[file_name]['strength'])
        config_yaml['a_prompt'] = cfg_dict[file_name]['a_prompt']
        config_yaml['n_prompt'] = cfg_dict[file_name]['n_prompt']
        config_yaml['sd_path'] = cfg_dict[file_name]['model']

    # inversion control
    if use_inv_prompts:
        config_yaml['inv_prompt'] = inv_pronpts_dict[name]
        inv_path_name = 'latents'
    else:
        config_yaml['inv_prompt'] = ''
        inv_path_name = 'latents-null'
    inv_latent_save_path = os.path.join(save_path, inv_path_name, file_name, f"inv_step_{config_yaml['inv_inference_steps']}")
    config_yaml['inv_save_path'] = inv_latent_save_path
    config_yaml['inv_latent_path'] = os.path.join(inv_latent_save_path, 'latents')
    
    # inversion noise produce
    if use_inv_noise and not os.path.exists(config_yaml['inv_latent_path']):
        os.makedirs(config_yaml['inv_latent_path'])
        with open(os.path.join(inv_latent_save_path, 'config.yaml'),'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        with open(os.path.join(save_path_video, 'config.yaml'),'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        os.system(f"conda activate fresco")
        subprocess.run(f"python {fresco_path}/preprocess.py {os.path.join(save_path_video, 'config.yaml')}", shell=True)
    
    # run all prompts
    for prompt in prompts:
        save_video_with_prompts = os.path.join(save_path_video, f"inv_step_{config_yaml['inv_inference_steps']}", 
                                               prompt.replace(' ', '_'), f"radix_{keyframe_select_radix}")
        os.makedirs(save_video_with_prompts, exist_ok=True)
        
        config_yaml['save_path'] = save_video_with_prompts +'/'
        config_yaml['prompt'] = prompt

        with open(os.path.join(save_video_with_prompts, 'config.yaml'),'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        
        subprocess.run(f"python {fresco_path}/run_fresco.py {os.path.join(save_video_with_prompts, 'config.yaml')}", shell=True)
