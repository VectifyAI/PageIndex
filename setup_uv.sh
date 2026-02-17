#!/bin/bash

# PageIndex UV 环境管理脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  PageIndex UV 环境管理${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  uv 未安装，正在安装...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo -e "${GREEN}✅ uv 安装完成${NC}"
fi

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}📦 创建虚拟环境...${NC}"
    uv venv
    echo -e "${GREEN}✅ 虚拟环境创建完成${NC}"
fi

# 激活虚拟环境并安装依赖
echo -e "${YELLOW}📥 安装项目依赖...${NC}"
uv pip install -e .
echo -e "${GREEN}✅ 依赖安装完成${NC}"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  未找到 .env 文件${NC}"
    echo -e "${BLUE}正在创建 .env 模板...${NC}"
    cat > .env << EOF
# OpenAI API Key
CHATGPT_API_KEY=your_openai_api_key_here

# 请将 your_openai_api_key_here 替换为您的真实 API Key
EOF
    echo -e "${GREEN}✅ .env 文件已创建，请编辑并添加您的 API Key${NC}"
else
    echo -e "${GREEN}✅ .env 文件已存在${NC}"
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}🎉 环境配置完成！${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "${YELLOW}使用方法：${NC}"
echo -e "  1. 激活虚拟环境："
echo -e "     ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo -e "  2. 运行 PageIndex："
echo -e "     ${GREEN}python run_pageindex.py --pdf_path /path/to/your/file.pdf${NC}"
echo ""
echo -e "  3. 退出虚拟环境："
echo -e "     ${GREEN}deactivate${NC}"
echo ""
