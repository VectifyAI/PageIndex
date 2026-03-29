"""Batch index Obsidian documents with PageIndex."""
import subprocess
import sys
import os
import time

OBSIDIAN = "E:/knowledge/Knowledge"

# Priority 0: Core index & architecture
P0 = [
    "00-Index/Home.md",
    "00-Index/Architecture.md",
    "00-Index/开发架构.md",
    "00-Index/开发系统.md",
    "00-Index/Roadmap.md",
    "00-Index/任务看板.md",
]

# Priority 1: Project specs
P1 = [
    "01-Projects/项目全景.md",
    "01-Projects/a2a-dev.md",
    "01-Projects/claw-mesh.md",
    "01-Projects/China MCP Gateway.md",
    "01-Projects/A股MCP-Adapter.md",
    "01-Projects/专家Agent-RAG.md",
    "01-Projects/CLAW Agent Lifecycle.md",
    "01-Projects/spec-as-infrastructure-plan.md",
    "01-Projects/PageIndex记忆升级.md",
    "01-Projects/memU迁移.md",
    "01-Projects/oh-my-claudecode.md",
    "01-Projects/OpenClaw.md",
    "01-Projects/Claw Router.md",
    "01-Projects/ClawAPI.md",
    "01-Projects/FSC.md",
    "01-Projects/NoPUA.md",
    "01-Projects/WinForge.md",
    "01-Projects/ZeroClaw.md",
    "01-Projects/agent-memory-middleware.md",
    "Projects/paperclip.md",
    "Projects/ane-analysis.md",
    "Projects/PantheonOS.md",
    "Projects/claw-mesh.md",
    "Projects/claw-router.md",
    "Projects/finshare.md",
    "Projects/FORGE.md",
    "Projects/cicada.md",
    "Projects/timo-reverse.md",
    "Projects/项目索引.md",
]

# Priority 2: Research, Infrastructure, Protocols, Trading
P2 = [
    "02-Infrastructure/记忆系统.md",
    "02-Infrastructure/SUPER-service-map.md",
    "02-Infrastructure/Memory-Nexus.md",
    "02-Infrastructure/NadirClaw-Integration-Report.md",
    "02-Infrastructure/集群运维.md",
    "02-Infrastructure/Costs.md",
    "02-Infrastructure/Incidents.md",
    "04-Research/Agent记忆系统全景.md",
    "04-Research/AI公司开源轮子全景.md",
    "04-Research/cli-anything-playbook.md",
    "04-Research/NVIDIA-AI-Blueprints-研究.md",
    "04-Research/specs-as-infrastructure.md",
    "04-Research/vibe-coding-critique.md",
    "04-Research/Agent-S-GUI自动化.md",
    "05-Protocols/Agent-Instructions.md",
    "05-Protocols/信息互通.md",
    "05-Protocols/复盘流程.md",
    "05-Protocols/Security-Audit.md",
    "03-Trading/AI自循环系统.md",
    "03-Trading/quant-terminal.md",
    "03-Trading/Failures.md",
]

def index_doc(rel_path):
    full_path = os.path.join(OBSIDIAN, rel_path)
    if not os.path.isfile(full_path):
        print(f"  SKIP (not found): {rel_path}")
        return False

    result_name = os.path.splitext(os.path.basename(rel_path))[0] + "_structure.json"
    result_path = os.path.join("results", result_name)
    if os.path.isfile(result_path):
        print(f"  SKIP (already indexed): {rel_path}")
        return True

    print(f"  Indexing: {rel_path} ...")
    python_exe = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
    try:
        proc = subprocess.run(
            [python_exe, "run_pageindex.py", "--md_path", full_path],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode == 0:
            print(f"  OK: {rel_path}")
            return True
        else:
            print(f"  FAIL: {rel_path}\n    {proc.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {rel_path}")
        return False

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    total = ok = fail = skip = 0
    for label, batch in [("P0-Core", P0), ("P1-Projects", P1), ("P2-Research", P2)]:
        print(f"\n{'='*50}")
        print(f"  {label} ({len(batch)} docs)")
        print(f"{'='*50}")
        for doc in batch:
            total += 1
            result = index_doc(doc)
            if result is True:
                ok += 1
            elif result is False:
                fail += 1
            # Brief pause to avoid rate limiting
            time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"  DONE: {ok}/{total} indexed, {fail} failed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
