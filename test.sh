OUTPUT_PATH=/mnt/netdisk/linjunxin/fresco

python preprocess.py $OUTPUT_PATH/output5-with-org-extra-prompt/car_red_input/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/car_red_input/inv_step_500/A_red_car_turns_in_the_winter/f_0.8_attn_0.5/config.yaml

python preprocess.py $OUTPUT_PATH/output5-with-org-extra-prompt/hamster_input/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/hamster_input/inv_step_500/A_hamster_walking_on_the_piano_in_cartoon_style/f_0.8_attn_0.5/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/hamster_input/inv_step_500/A_white_cat_walking_on_the_piano_in_cartoon_style/f_0.8_attn_0.5/config.yaml

python preprocess.py $OUTPUT_PATH/output5-with-org-extra-prompt/panther_input/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/panther_input/inv_step_500/A_black_panther_in_CG_style/f_0.8_attn_0.5/config.yaml

python preprocess.py $OUTPUT_PATH/output5-with-org-extra-prompt/sakura_input/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/sakura_input/inv_step_500/Cartoon_Haruno_Sakura_looking_in_the_railway_station,_flat,_2D_/f_0.8_attn_0.5/config.yaml


python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/woman_cg_input/inv_step_500/A_beautiful_woman_in_CG_style/f_0.8_attn_0.7/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/paladin_input/inv_step_500/A_paladin_walks_in_the_dark_forest_carrying_a_holy_sword/f_0.8_attn_0.7/config.yaml
python run_fresco_ddim.py $OUTPUT_PATH/output5-with-org-extra-prompt/cat_white_input/inv_step_500/A_white_cat_in_pink_background/f_0.8_attn_0.7/config.yaml

