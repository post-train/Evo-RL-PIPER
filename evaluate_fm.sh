# 中文简介：离线评估某个 Flow Matching checkpoint 的动作拟合效果，用于选模型。

python /home/szk/szk/Evo-RL/src/lerobot/scripts/evaluate_fm_action_fit.py \
  --checkpoint /home/szk/szk/Evo-RL/outputs/flow_matching_train_20260501_004135/checkpoints/080000/pretrained_model \
  --device cuda \
  --output /home/szk/szk/Evo-RL/outputs/flow_matching_80k_start_fit_eval3.json
