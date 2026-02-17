#!/bin/bash

# 多模型快速切换脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  PageIndex 模型切换工具${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 模型配置
declare -A MODELS=(
    # 国内模型
    ["1"]="通义千问 qwen-max|sk-eb7aaab9089449e9a1a54207f6157d84|https://dashscope.aliyuncs.com/compatible-mode/v1|qwen-max"
    ["2"]="DeepSeek deepseek-chat|your-deepseek-api-key|https://api.deepseek.com|deepseek-chat"
    ["3"]="智谱 GLM-4 Plus|your-glm-api-key|https://open.bigmodel.cn/api/paas/v4|glm-4-plus"
    ["4"]="Kimi moonshot-v1-32k|your-kimi-api-key|https://api.moonshot.cn/v1|moonshot-v1-32k"
    ["5"]="硅基流动 Qwen2.5|your-siliconflow-api-key|https://api.siliconflow.cn/v1|Qwen/Qwen2.5-72B-Instruct"
    ["6"]="零一万物 Yi-Large|your-yi-api-key|https://api.lingyiwanwu.com/v1|yi-large"
    ["7"]="MiniMax abab6.5|your-minimax-api-key|https://api.minimax.chat/v1|abab6.5-chat"
    # 国外模型
    ["11"]="OpenAI GPT-4o|your-openai-api-key||gpt-4o-2024-11-20"
    ["12"]="Anthropic Claude 3.5|your-anthropic-api-key|https://api.anthropic.com/v1|claude-3-5-sonnet-20241022"
    ["13"]="Google Gemini 2.0|your-gemini-api-key|https://generativelanguage.googleapis.com/v1beta/openai|gemini-2.0-flash-exp"
    ["14"]="Groq Llama 3.3|your-groq-api-key|https://api.groq.com/openai/v1|llama-3.3-70b-versatile"
    ["15"]="OpenRouter|your-openrouter-api-key|https://openrouter.ai/api/v1|anthropic/claude-3.5-sonnet"
    # 本地部署
    ["21"]="Ollama 本地|ollama|http://localhost:11434/v1|llama3.1:70b"
    ["22"]="LM Studio 本地|lm-studio|http://localhost:1234/v1|local-model"
)

echo -e "${YELLOW}请选择要切换的模型：${NC}"
echo ""
echo "=== 国内模型 ==="
echo "1) 通义千问 qwen-max（当前配置）"
echo "2) DeepSeek deepseek-chat（性价比高 💰）"
echo "3) 智谱 GLM-4 Plus（中文优秀）"
echo "4) Kimi moonshot-v1-32k（超长上下文）"
echo "5) 硅基流动 Qwen2.5（速度快）"
echo "6) 零一万物 Yi-Large（中文能力强）"
echo "7) MiniMax abab6.5（多模态）"
echo ""
echo "=== 国外模型 ==="
echo "11) OpenAI GPT-4o（性能最强）"
echo "12) Anthropic Claude 3.5 Sonnet（推理强）"
echo "13) Google Gemini 2.0 Flash（多模态）"
echo "14) Groq Llama 3.3 70B（极速推理 ⚡）"
echo "15) OpenRouter（聚合平台，一键访问所有模型）"
echo ""
echo "=== 本地部署 ==="
echo "21) Ollama（本地部署，完全免费）"
echo "22) LM Studio（图形界面）"
echo ""
echo "=== 其他 ==="
echo "99) 自定义配置"
echo "0) 退出"
echo ""

read -p "请输入选项: " choice

if [ "$choice" = "0" ]; then
    echo -e "${GREEN}退出${NC}"
    exit 0
fi

if [ "$choice" = "99" ]; then
    echo ""
    echo -e "${YELLOW}自定义配置${NC}"
    read -p "API Key: " api_key
    read -p "API Base URL (留空使用 OpenAI 官方): " api_base
    read -p "模型名称: " model_name
else
    if [ -z "${MODELS[$choice]}" ]; then
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
    fi
    
    IFS='|' read -r name api_key api_base model_name <<< "${MODELS[$choice]}"
    echo ""
    echo -e "${BLUE}选择的模型: ${name}${NC}"
    
    # 检查是否需要输入 API Key
    if [[ "$api_key" == "your-"* ]]; then
        echo -e "${YELLOW}⚠️  需要配置 API Key${NC}"
        read -p "请输入您的 API Key: " api_key
    fi
fi

# 备份当前 .env 文件
if [ -f ".env" ]; then
    cp .env .env.backup
    echo -e "${GREEN}✅ 已备份当前配置到 .env.backup${NC}"
fi

# 写入新配置
cat > .env << EOF
# ============================================
# LLM API 配置
# ============================================
# 当前使用：${model_name}
# 配置时间：$(date '+%Y-%m-%d %H:%M:%S')

# API Key（必填）
CHATGPT_API_KEY=${api_key}

EOF

if [ -n "$api_base" ]; then
    cat >> .env << EOF
# API 基础 URL
OPENAI_API_BASE=${api_base}

EOF
fi

cat >> .env << EOF
# 默认模型
OPENAI_MODEL=${model_name}

# ============================================
# 说明
# ============================================
# 如需切换模型，请运行: ./switch_model.sh
# 恢复备份配置: cp .env.backup .env
EOF

echo ""
echo -e "${GREEN}✅ 配置已更新${NC}"
echo ""
echo -e "${BLUE}新配置：${NC}"
echo "  API Key: ${api_key:0:20}..."
if [ -n "$api_base" ]; then
    echo "  API Base: ${api_base}"
else
    echo "  API Base: OpenAI 官方"
fi
echo "  Model: ${model_name}"
echo ""
echo -e "${YELLOW}测试配置：${NC}"
echo "  source .venv/bin/activate"
echo "  python test_qwen_api.py"
echo ""
echo -e "${YELLOW}开始使用：${NC}"
echo "  python run_pageindex.py --pdf_path your_file.pdf"
echo ""
