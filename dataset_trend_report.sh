# 中文简介：说明并指向动作趋势分析脚本，用来查看数据集里动作变化强度与未来趋势统计。

# 脚本已经加好了：src/lerobot/scripts/lerobot_action_trend_report.py:1

#   它会输出这些重点统计：

#   - 全局动作变化强度：相邻帧动作 L2、动作与状态的距离
#   - 未来趋势强度：state(t) 到 action(t+k) 的 k=1..N 距离
#   - 窗口内趋势弱不弱：未来 n_action_steps 窗口的平均 joint std
#   - 静态保持基线：把当前 state 原样保持未来 8 步时的 raw/normalized L1
#   - 逐关节统计：action_std、mean_abs_delta、adjacent_zero_delta_ratio
#   - 最静态 / 最动态 episode 排行

#   我没有执行。你直接跑下面的命令就行。

#   文本报告：

#   python src/lerobot/scripts/lerobot_action_trend_report.py \
#     --dataset /home/szk/szk/Evo-RL/datasets3 \
#     --n-obs-steps 8 \
#     --n-action-steps 8 \
#     --max-ahead 8

#   JSON 报告：

#   python src/lerobot/scripts/lerobot_action_trend_report.py \
#     --dataset /home/szk/szk/Evo-RL/datasets3 \
#     --n-obs-steps 8 \
#     --n-action-steps 8 \
#     --max-ahead 8 \
#     --json

#   如果你想把结果落盘：

#   python src/lerobot/scripts/lerobot_action_trend_report.py \
#     --dataset /home/szk/szk/Evo-RL/datasets3 \
#     --n-obs-steps 8 \
#     --n-action-steps 8 \
#     --max-ahead 8 \
#     > /home/szk/szk/Evo-RL/outputs/datasets3_action_trend_report.txt

  python src/lerobot/scripts/lerobot_action_trend_report.py \
    --dataset /home/szk/szk/Evo-RL/datasets3 \
    --n-obs-steps 8 \
    --n-action-steps 8 \
    --max-ahead 8
