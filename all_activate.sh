#!/bin/bash

# 中文简介：批量拉起并配置本机的 CAN 接口，方便双臂 Piper 设备通信前统一初始化。

# 定义要激活的接口列表
INTERFACES=("can0" "can1" "can2" "can3")
# 设置位速率 (1.0 Mbps)
BITRATE=1000000

echo "============================================"
echo "正在配置 SocketCAN 接口 (经典模式 1Mbps)..."
echo "============================================"

for iface in "${INTERFACES[@]}"; do
    # 检查接口是否存在
    if ip link show "$iface" > /dev/null 2>&1; then
        echo "正在设置 $iface..."
        
        # 先关闭接口，防止配置冲突
        sudo ip link set "$iface" down
        
        # 激活接口：设置类型为 can，速率为 1M，不开启 FD
        sudo ip link set "$iface" up type can bitrate $BITRATE
        
        # 检查是否成功
        if [ $? -eq 0 ]; then
            echo " [✓] $iface 已启动 (MTU: $(cat /sys/class/net/$iface/mtu))"
        else
            echo " [✗] $iface 启动失败"
        fi
    else
        echo " [-] 跳过 $iface: 系统中未发现该硬件"
    fi
done

echo "--------------------------------------------"
echo "当前 CAN 接口状态汇总："
ip link show | grep -E "can[0-3]"
echo "============================================"
