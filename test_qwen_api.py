#!/usr/bin/env python3
"""
通义千问 API 连接测试脚本
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_api_connection():
    """测试通义千问 API 连接"""
    print("🧪 开始测试通义千问 API 连接...")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv("CHATGPT_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL", "qwen-max")
    
    if not api_key:
        print("❌ 错误: 未找到 CHATGPT_API_KEY 环境变量")
        print("请检查 .env 文件配置")
        sys.exit(1)
    
    print(f"✅ API Key: {api_key[:20]}...")
    print(f"✅ API Base: {api_base}")
    print(f"✅ Model: {model}")
    print("=" * 60)
    
    # 测试 API 调用
    try:
        from pageindex.utils import ChatGPT_API
        
        print("\n📡 发送测试请求...")
        response = ChatGPT_API(
            model=model,
            prompt="你好！请用一句话介绍你自己。"
        )
        
        print("\n✅ API 调用成功！")
        print("=" * 60)
        print("📝 响应内容:")
        print(response)
        print("=" * 60)
        print("\n🎉 通义千问 API 配置正确，可以正常使用！")
        
    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        print("\n请检查:")
        print("1. API Key 是否正确")
        print("2. API Key 是否已激活")
        print("3. 账户余额是否充足")
        print("4. 网络连接是否正常")
        sys.exit(1)

if __name__ == "__main__":
    test_api_connection()
