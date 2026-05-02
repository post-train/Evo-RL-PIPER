#!/bin/bash

# 中文简介：检查当前机器上的 CAN 设备和 udev 规则是否配置正确，用于排查 CAN 口识别问题。

echo "=== 检查 CAN 设备 ==="
ip link show | grep -E "can[0-3]"

echo -e "\n=== 检查 udev 规则文件 ==="
if [ -f /etc/udev/rules.d/99-usb-can.rules ]; then
    echo "规则文件存在:"
    cat /etc/udev/rules.d/99-usb-can.rules
else
    echo "规则文件不存在!"
    exit 1
fi

echo -e "\n=== 序列号对比检查 ==="

# 定义期望的序列号（从 udev 规则中提取）
declare -A expected_serials=(
    ["can0"]="001E00385443571020393433"
    ["can1"]="004C003A5443570F20393433"
    ["can2"]="001E00225631511820313857"
    ["can3"]="004600205631511820313857"
)

mismatch_count=0

for can in can0 can1 can2 can3; do
    echo "--- $can ---"
    
    # 获取实际序列号（取第一个匹配的 serial）
    actual_serial=$(udevadm info -a /sys/class/net/$can 2>/dev/null | grep "ATTRS{serial}==" | head -1 | sed 's/.*ATTRS{serial}=="\([^"]*\)".*/\1/')
    expected_serial=${expected_serials[$can]}
    
    echo "  规则期望: $expected_serial"
    echo "  实际设备: $actual_serial"
    
    if [ "$actual_serial" = "$expected_serial" ]; then
        echo "  状态: ✅ 匹配"
    else
        echo "  状态: ❌ 不匹配"
        mismatch_count=$((mismatch_count + 1))
    fi
done

echo -e "\n=== 检查结果 ==="
if [ $mismatch_count -eq 0 ]; then
    echo "所有设备序列号匹配，udev 规则正确！"
else
    echo "发现 $mismatch_count 个设备不匹配！"
    echo -e "\n建议更新 /etc/udev/rules.d/99-usb-can.rules："
    echo "复制以下内容到规则文件："
    echo ""
    for can in can0 can1 can2 can3; do
        actual_serial=$(udevadm info -a /sys/class/net/$can 2>/dev/null | grep "ATTRS{serial}==" | head -1 | sed 's/.*ATTRS{serial}=="\([^"]*\)".*/\1/')
        echo "# $can"
        echo "SUBSYSTEM==\"net\", ACTION==\"add\", ATTRS{serial}==\"$actual_serial\", NAME=\"$can\""
        echo ""
    done
fi
