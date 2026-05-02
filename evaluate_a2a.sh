# 中文简介：离线评估某个 Original A2A checkpoint 在数据集起始片段上的动作拟合效果。

python /home/szk/szk/Evo-RL/src/lerobot/scripts/evaluate_a2a_action_fit.py \
  --checkpoint /home/szk/szk/Evo-RL/outputs/original_a2a_train_20260428_172210/checkpoints/040000/pretrained_model \
  --device cuda \
  --output /home/szk/szk/Evo-RL/outputs/original_a2a_40k_start_fit_eval2.json
