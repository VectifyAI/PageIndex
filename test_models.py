#!/usr/bin/env python3
"""
多模型对比测试脚本
比较不同模型在相同任务上的表现
"""

import os
import sys
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_model(model_name, prompt):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    try:
        from pageindex.utils import ChatGPT_API
        
        start_time = time.time()
        response = ChatGPT_API(model=model_name, prompt=prompt)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        print(f"✅ 响应成功")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
        print(f"📝 响应内容:")
        print(f"{response[:200]}..." if len(response) > 200 else response)
        
        return {
            "model": model_name,
            "success": True,
            "time": elapsed_time,
            "response_length": len(response),
            "response": response
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }

def main():
    """主函数"""
    print("🧪 PageIndex 多模型对比测试")
    print("="*60)
    
    # 测试提示词
    prompt = """请用一句话总结以下文本的主要内容：

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"""
    
    # 要测试的模型列表
    models = [
        "qwen-max",           # 通义千问
        "deepseek-chat",      # DeepSeek
        "glm-4-plus",         # 智谱 GLM
        "moonshot-v1-32k",    # Kimi
        "gpt-4o-2024-11-20",  # OpenAI
    ]
    
    print(f"\n📋 测试提示词:")
    print(f"{prompt[:100]}...")
    print(f"\n🎯 将测试 {len(models)} 个模型")
    
    # 测试所有模型
    results = []
    for model in models:
        result = test_model(model, prompt)
        results.append(result)
        time.sleep(1)  # 避免请求过快
    
    # 输出对比结果
    print(f"\n\n{'='*60}")
    print("📊 测试结果对比")
    print(f"{'='*60}\n")
    
    print(f"{'模型':<25} {'状态':<8} {'耗时(秒)':<12} {'响应长度':<10}")
    print("-" * 60)
    
    for result in results:
        if result["success"]:
            status = "✅ 成功"
            time_str = f"{result['time']:.2f}"
            length_str = str(result['response_length'])
        else:
            status = "❌ 失败"
            time_str = "-"
            length_str = "-"
        
        print(f"{result['model']:<25} {status:<8} {time_str:<12} {length_str:<10}")
    
    # 找出最快的模型
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x["time"])
        print(f"\n🏆 最快模型: {fastest['model']} ({fastest['time']:.2f}秒)")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("CHATGPT_API_KEY"):
        print("❌ 错误: 未找到 CHATGPT_API_KEY 环境变量")
        print("请先配置 .env 文件")
        sys.exit(1)
    
    main()
