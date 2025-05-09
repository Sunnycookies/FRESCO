import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
import cv2
from PIL import Image
from run_fresco import run_full_video_translation

indexes_map = ['0000.png', '0029.png', '0040.png', '0052.png', 
               '0063.png', '0073.png', '0084.png', '0098.png', 
               '0108.png', '0123.png', '0137.png', '0148.png', 
               '0161.png', '0172.png', '0182.png', '0196.png', 
               '0207.png', '0218.png', '0233.png', '0243.png', 
               '0268.png', '0279.png', '0294.png', '0304.png', 
               '0317.png', '0334.png', '0345.png', '0358.png', 
               '0368.png', '0378.png', '0389.png', '0399.png']
keys = []
for i in indexes_map:
    keys.append(int(i.split('.')[0]))

def extract_frames(root):
    parent_path = os.path.dirname(root)
    image_path = os.path.join(parent_path, 'keys')
    os.system(f"rm -rf {image_path}; mkdir {image_path}")

    cap = cv2.VideoCapture(root)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(n_frame):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame).save(os.path.join(image_path, '%04d.png'%(i)))

def ebsynth_run(target_dir):
    root = os.path.join(target_dir, 'truncated_pnp_Tokenflow_loop.mp4')
    extract_frames(root)
    
    key_dir = os.path.join(target_dir, 'keys')
    indexes = os.listdir(key_dir)
    indexes.sort(reverse=True)
    print(indexes)
    
    for index in indexes:
        idx = int(index.split('.')[0])
        if idx > len(indexes_map):
            print('already mapped')
            break
        print(index, idx, indexes_map[idx])
        os.system(f"mv {os.path.join(key_dir, index)} {os.path.join(key_dir, indexes_map[idx])}")
    
    config_p = os.path.join(target_dir, 'config.yaml')
    with open(config_p, "r") as f:
        cfg = yaml.safe_load(f)

    cfg['synth_mode'] = 'Mixed'
    cfg['file_path'] = '/home/color/fresco_new/data/woman_input.mp4'

    run_full_video_translation(cfg, keys, True)
    
dirs = [
    # '/mnt/netdisk/baiyh/fresco_new/test-Tokenflow-pnp/truncated-woman-32-sd2.1/inv_step_500/a_shiny_silver_robotic_woman,_futuristic/radix_4',
    '/mnt/netdisk/baiyh/fresco_new/test-Tokenflow-pnp/truncated-woman-32-sd2.1/inv_step_500/a_photo_of_a_wooden_statue/radix_4',
    # '/mnt/netdisk/baiyh/fresco_new/test-Tokenflow-pnp/truncated-woman-32-sd2.1/inv_step_500/a_handsome_man_in_Assassin_Creed/radix_4',
    # '/mnt/netdisk/baiyh/fresco_new/test-Tokenflow-pnp/truncated-woman-32-sd2.1/inv_step_500/a_close_up_painting_of_Von_Gogh_with_Cloak/radix_4'
        ]
for tgt_dir in dirs:
    print(f"target dir: {tgt_dir}")
    ebsynth_run(tgt_dir)