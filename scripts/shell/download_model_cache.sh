#!/bin/sh
SCRIPT_DIR=$(dirname "$(realpath "$0")")
# # 定义SAS令牌文件路径（与脚本同目录）
SAS_FILE="${SCRIPT_DIR}/sas.txt"

# 读取SAS令牌并处理特殊字符
SAS_TOKEN=$(tr -d '\n' < "$SAS_FILE" | sed 's/[\"\\]/\\&/g')

# 你的固定URL（请修改为实际URL）
BASE_URL="https://igshare.blob.core.windows.net/zelonglv/model_cache/hub/"

/tmp/azcopy copy "${BASE_URL}${SAS_TOKEN}" $1 --recursive