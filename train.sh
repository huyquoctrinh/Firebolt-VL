# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_ddp.py

# # CUDA_VISIBLE_DEVICES=1 python transfer_weights_simple.py \
# #     --viper_checkpoint /home/mamba/ML_project/Testing/Huy/joint_vlm/results_viper_f1_s2_s4_ssm_topk4_ablation_cot_reasoning/epoch_2/ \
# #     --output_dir /home/mamba/ML_project/Testing/Huy/joint_vlm/firebolt_vl_v0/

CUDA_VISIBLE_DEVICES=0 python infer.py