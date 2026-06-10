"""
言犀 AI 智能通话助手 — 主入口
==============================
文本模式 / 语音模式 / 测试模式。

运行方式:
    python src/main.py               # 文本模式（默认）
    python src/main.py --test        # 测试模式
"""

import argparse
import os
import sys
from pathlib import Path

# 计算项目根目录（在导入其他模块之前）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env 文件（必须在其他导入之前）
from dotenv import load_dotenv
env_path = _PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] 已加载环境变量: {env_path}")

# Windows 中文环境强制 UTF-8 输出
if sys.platform == "win32":
    try:
        import io
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        else:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8"
            )
    except Exception:
        pass

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(_PROJECT_ROOT))

from src.orchestration.orchestrator import CallOrchestrator
from src.core.config import load_and_validate_config
from src.utils.logger import get_logger, configure_logging

logger = get_logger(__name__)


def print_banner():
    """打印欢迎横幅"""
    banner = r"""
╔════════════════════════════════════════════════════════════╗
║          言犀 - AI 智能通话助手 整合版 v1.0                   ║
║                                                            ║
║  [三路检索] 向量语义 + 知识图谱 + n-gram 精确匹配            ║
║  [LLM抽象]  DeepSeek / 智谱GLM / 通义千问 统一切换          ║
║  [诈骗检测] -> 拦截拒接                                      ║
║  [业务来电] -> 对话 -> 信息卡片                               ║
║  [紧急来电] -> 转接机主                                      ║
║                                                            ║
║  输入 /help 查看所有命令                                     ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
命令帮助:
  来电模拟:
    直接输入文字模拟来电内容
    格式: [号码] 内容
    示例: 13800001111 我是美团外卖的

  系统命令:
    /help         显示帮助
    /status       显示系统状态
    /presence     设置机主状态 (busy/free/dnd)
    /quit         退出
"""
    print(help_text)


def run_text_mode(orchestrator: CallOrchestrator):
    """文本交互模式"""
    print("\n文本模式已启动（输入 /help 查看命令）\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/quit":
            print("再见！")
            break
        if user_input == "/status":
            status = {
                "LLM后端": orchestrator.llm_client.backend.value,
                "可用后端": orchestrator.llm_client.available_backends,
                "BM25检索": orchestrator.hybrid_retriever.bm25.is_available(),
                "向量检索": orchestrator.hybrid_retriever.vector.is_available(),
                "混合检索": orchestrator.hybrid_retriever.is_available(),
                "机主状态": orchestrator.presence_mode,
            }
            for k, v in status.items():
                print(f"  {k}: {v}")
            continue
        if user_input.startswith("/presence "):
            mode = user_input[10:].strip()
            orchestrator.set_presence(mode)
            print(f"  机主状态已设为: {mode}")
            continue

        # 解析来电
        import re
        match = re.match(r'^(\d{5,15})\s+(.+)$', user_input)
        if match:
            caller_number, call_text = match.group(1), match.group(2)
        else:
            caller_number, call_text = "", user_input

        # 处理来电
        result = orchestrator.run(call_text, caller_number=caller_number)

        # 显示结果
        print(f"\n[结果] {result.get('call_type_name', '?')} "
              f"(置信度={result.get('confidence', 0):.0%})")
        print(f"  动作: {result.get('final_action', '?')}")
        if result.get('agent_reply'):
            print(f"  言犀: {result['agent_reply']}")
        if result.get('notification_card'):
            card = result['notification_card']
            print(f"  卡片: {card.get('title', '')} - {card.get('body', '')[:40]}")
        print()


def run_test_mode(orchestrator: CallOrchestrator):
    """测试模式"""
    print("\n运行测试用例...\n")

    tests = [
        ("13800001111", "我是美团外卖的，你的餐到楼下了", "food_delivery"),
        ("13800002222", "您好，我是市公安局的，你涉嫌洗钱案件", "scam"),
        ("13800003333", "你的快递到菜鸟驿站了", "express"),
        ("13800004444", "妈，我今天晚上回家吃饭", "family"),
        ("13800005555", "领导，明天下午有个紧急会议", "leader"),
        ("13800002222", "你好，我是检察院的", "scam"),
        ("13800006666", "恭喜您中奖了！奖金10万元，请先缴纳个人所得税", "scam"),
    ]

    for number, text, expected in tests:
        result = orchestrator.run(text, caller_number=number)
        actual = result.get('type_id', '?')
        action = result.get('final_action', '?')
        reply = result.get('agent_reply', '')[:50]
        card = result.get('notification_card')
        card_info = f'[卡片] {card["title"]}' if card else ''

        match = '[OK]' if expected in actual or actual in expected else '[?]'
        print(f"{match} [{number}] {text[:30]}...")
        print(f"   分类: {actual} | 动作: {action} | {card_info}")
        print(f"   回复: {reply}")
        print()

    print("测试完成！")


def main():
    parser = argparse.ArgumentParser(description="言犀 — AI 智能通话助手 整合版")
    parser.add_argument("--text", action="store_true", help="文本模式（默认）")
    parser.add_argument("--test", action="store_true", help="测试模式")
    args = parser.parse_args()

    print_banner()

    # 加载配置
    logger.info("正在加载配置...")
    config = load_and_validate_config("config.yaml")
    configure_logging(config)

    # 初始化调度器
    logger.info("正在初始化调度器...")
    try:
        orchestrator = CallOrchestrator(config)
        logger.info("所有模块初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 选择运行模式
    if args.test:
        run_test_mode(orchestrator)
    else:
        run_text_mode(orchestrator)


if __name__ == "__main__":
    main()
