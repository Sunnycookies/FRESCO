import os
import yaml
import json
import argparse
#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# manually set parameters
save_path = '/mnt/netdisk/baiyh/fresco_new'
default_sd = 'stabilityai/stable-diffusion-2-1-base'
default_controlnet_type = 'hed'
use_saliency = False
use_freeu = False
cond_scale = 0.7
batch_size = 8
num_inference_steps = 50
num_warmup_steps = 0
end_opt_step = 40
max_process = 4
pnp_attn_t = 0.6
pnp_f_t = 0.8
inv_inference_steps = 500
inv_batch_size = 20

# commandline parameters
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edit', type=str, 
                    default='pnp', choices=['pnp', 'SDEdit'],
                    help='key frame edit method')
parser.add_argument('-s', '--synth', type=str, 
                    default='Tokenflow', choices=['Tokenflow', 'ebsynth', 'None', 'Mixed'],
                    help='non-key frame synthesis mode')
parser.add_argument('-m', '--mirror', 
                    action='store_true',
                    help='set to use mirror site to download models and data')
parser.add_argument('-f', '--fresco', 
                    action='store_true',
                    help='set to use FRESCO')
parser.add_argument('-km', '--keymode', type=str, 
                    default='loop', choices=['loop',' fixed'],
                    help='key frame selection mode')
parser.add_argument('-kr', '--keyradix', type=int, 
                    default=4,
                    help='key frame selection radix')
args = parser.parse_args()

edit_method = args.edit
synth_method = args.synth
use_fresco = args.fresco
use_mirror = args.mirror
keyframe_select_mode = args.keymode
keyframe_select_radix = args.keyradix

use_inversion = edit_method == 'pnp' or synth_method in ['Tokenflow', 'Mixed']
if use_mirror:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# prepare data
fresco_path = os.getcwd()

# video names end with 'input' are SDEdit data, and those end with integer are pnp data
video_dir = os.path.join(fresco_path, 'data')
video_list = [
    # 'bread-80',
    # 'basketball-man-120', 
    'tesla-200',
    ]    
    
config_paras = os.path.join(fresco_path, 'cfg.json')
with open(config_paras, 'r') as f:
    cfg_dict = json.load(f)

# check if models have been downloaded
if not os.path.exists(os.path.join(fresco_path, 'model')):
    os.system(f"python {os.path.join(fresco_path, 'install.py')}{' --mirror' if use_mirror else''}")
gmflow_path = os.path.join(fresco_path, 'model', 'gmflow_sintel-0c07dcb3.pth')
sod_path = os.path.join(fresco_path, 'model', 'epoch_resnet.pth')

# make and write configuration file
def make_config(file_name, prompt = None):
    config_yaml = {}
    
    # supporting models
    config_yaml['gmflow_path'] = gmflow_path
    config_yaml['sod_path'] = sod_path
    config_yaml['use_saliency'] = use_saliency

    # methods
    config_yaml['edit_mode'] = edit_method
    config_yaml['synth_mode'] = synth_method
    config_yaml['use_fresco'] = use_fresco
    
    # keyframe selection
    config_yaml['mininterv'] = cfg_dict[file_name]['mininterv']
    config_yaml['maxinterv'] = cfg_dict[file_name]['maxinterv']
    config_yaml['keyframe_select_mode'] = keyframe_select_mode
    if synth_method not in ['Tokenflow', 'Mixed']:
        config_yaml['keyframe_select_mode'] = 'fixed'
    config_yaml['keyframe_select_radix'] = keyframe_select_radix
    config_yaml['primary_select'] = synth_method == 'Mixed'
    if synth_method == 'None':
        config_yaml['mininterv'] = config_yaml['maxinterv'] = 1

    # data
    suffix = f"test-{synth_method}-{edit_method}"
    suffix += f"{'-no_fresco' if not use_fresco else ''}"
    video_save_path = os.path.join(save_path, suffix, file_name)
    prompt = cfg_dict[file_name]['prompt'] if prompt is None else prompt
    save_path_with_prompts = os.path.join(video_save_path, f"inv_step_{inv_inference_steps}", 
                                          prompt.replace(' ', '_'), f"radix_{keyframe_select_radix}")
    os.makedirs(save_path_with_prompts, exist_ok=True)
    config_yaml['file_path'] = os.path.join(video_dir, f"{file_name}.mp4")
    config_yaml['save_path'] = save_path_with_prompts
    
    # diffusion
    config_yaml['sd_path'] = cfg_dict[file_name]['model']
    # config_yaml['seed'] = int(edit_method != 'SDEdit')
    config_yaml['seed'] = 0
    config_yaml['use_controlnet'] = edit_method == 'SDEdit'
    config_yaml['cond_scale'] = cfg_dict[file_name]['control_scales']
    config_yaml['controlnet_type'] = cfg_dict[file_name]['control']
    config_yaml['use_freeu'] = use_freeu
    config_yaml['prompt'] = prompt
    config_yaml['a_prompt'] = cfg_dict[file_name]['a_prompt']
    config_yaml['n_prompt'] = cfg_dict[file_name]['n_prompt']
    
    # video2video translation
    config_yaml['batch_size'] = batch_size
    config_yaml['num_inference_steps'] = num_inference_steps
    config_yaml['num_warmup_steps'] = int(config_yaml['num_inference_steps'] * cfg_dict[file_name]['strength'])
    config_yaml['end_opt_step'] = end_opt_step
    config_yaml['max_process'] = max_process
    config_yaml['num_intraattn_steps'] = int(edit_method != 'pnp')
    config_yaml['pnp_attn_t'] = pnp_attn_t
    config_yaml['pnp_f_t'] = pnp_f_t
    config_yaml['temp_paras_save_path'] = os.path.join(fresco_path, 'paras')

    # inversion
    if use_inversion:
        config_yaml['inv_prompt'] = cfg_dict[file_name]['inv_prompt']
        latent_path_name = 'latents'
    else:
        config_yaml['inv_prompt'] = ''
        latent_path_name = 'latents-null'
    inv_save_path = os.path.join(save_path, latent_path_name, file_name, f"inv_step_{inv_inference_steps}")
    inv_latent_path = os.path.join(inv_save_path, 'latents')
    config_yaml['inv_save_path'] = inv_save_path
    config_yaml['inv_latent_path'] = inv_latent_path
    config_yaml['use_inversion'] = use_inversion
    config_yaml['inv_inference_steps'] = inv_inference_steps
    config_yaml['inv_batch_size'] = inv_batch_size
    
    # write configuration file
    config_path = os.path.join(save_path_with_prompts, 'config.yaml')
    with open(config_path,'w') as f:
        yaml.dump(config_yaml, f, default_flow_style=False)
    
    # inversion noise preprocess
    if use_inversion and (not os.path.exists(inv_latent_path) or len(os.listdir(inv_latent_path)) == 0):
        os.makedirs(inv_latent_path, exist_ok=True)
        with open(os.path.join(inv_save_path, 'config.yaml'),'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        with open(os.path.join(video_save_path, 'config.yaml'),'w') as f:
            yaml.dump(config_yaml, f, default_flow_style=False)
        os.system(f"python {os.path.join(fresco_path, 'preprocess.py')} {os.path.join(video_save_path, 'config.yaml')}")
    
    return config_path

# produce configuration and run video2video translation
for file_name in cfg_dict.keys():
    # produce certain video translation configurations
    if file_name not in video_list:
        continue
    
    # produce configurations and run video2video translation
    config_path = make_config(file_name)
    # print(f"python {os.path.join(fresco_path, 'run_fresco.py')} {config_path}")
    os.system(f"python {os.path.join(fresco_path, 'run_fresco.py')} {config_path}")

