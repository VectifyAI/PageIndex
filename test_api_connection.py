#!/usr/bin/env python3
"""
API 连接测试脚本
用于诊断 API 连接问题
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen-max")

def test_api_connection():
    print("=" * 50)
    print("API 连接测试")
    print("=" * 50)
    print(f"API Base URL: {OPENAI_API_BASE}")
    print(f"API Key: {CHATGPT_API_KEY[:10]}...{CHATGPT_API_KEY[-4:]}" if CHATGPT_API_KEY else "未设置")
    print(f"模型: {OPENAI_MODEL}")
    print("=" * 50)

    if not CHATGPT_API_KEY:
        print("错误: CHATGPT_API_KEY 未设置")
        return

    # 创建客户端
    if OPENAI_API_BASE:
        client = openai.OpenAI(api_key=CHATGPT_API_KEY, base_url=OPENAI_API_BASE)
    else:
        client = openai.OpenAI(api_key=CHATGPT_API_KEY)

    # 测试不同模型
    models_to_test = ["qwen-max", "qwen-plus", "qwen-turbo", "qwen-long"]

    for model in models_to_test:
        print(f"\n测试模型: {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "你好，请回复'API连接成功'"}],
                temperature=0,
                max_tokens=20
            )
            print(f"  ✅ {model} 连接成功!")
            print(f"     响应: {response.choices[0].message.content}")
        except openai.AuthenticationError as e:
            print(f"  ❌ {model} 认证失败: API Key 无效")
            print(f"     错误详情: {e}")
        except openai.PermissionDeniedError as e:
            print(f"  ❌ {model} 权限被拒绝: 账户可能没有该模型的访问权限")
            print(f"     错误详情: {e}")
            print(f"     建议: 请登录阿里云控制台检查是否已开通该模型")
        except openai.RateLimitError as e:
            print(f"  ⚠️ {model} 达到速率限制")
            print(f"     错误详情: {e}")
        except openai.APIError as e:
            print(f"  ❌ {model} API错误: {e}")
        except Exception as e:
            print(f"  ❌ {model} 未知错误: {e}")

    print("\n" + "=" * 50)
    print("诊断建议:")
    print("=" * 50)
    print("""
1. 如果所有模型都显示 '权限被拒绝':
   - 请访问 https://dashscope.console.aliyun.com/apiKey 检查 API Key 是否有效
   - 检查账户是否完成实名认证
   - 检查账户余额是否充足

2. 如果只有 qwen-max 失败:
   - qwen-max 是高级模型，可能需要额外开通
   - 尝试在 config.yaml 中切换到 qwen-plus 或 qwen-turbo

3. 如果需要重新生成 API Key:
   - 访问 https://dashscope.console.aliyun.com/apiKey
   - 创建新的 API Key 并更新 .env 文件
""")

if __name__ == "__main__":
    test_api_connection()
