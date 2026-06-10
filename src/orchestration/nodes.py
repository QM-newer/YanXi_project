"""
Agent 节点函数
==============
四个独立 Agent 函数：classify_intent, handle_takeout, handle_important, handle_scam。

每个函数统一签名: (state: OrchestratorState) -> dict
核心逻辑与交互完全解耦，不包含 input()/print()。
"""

from src.orchestration.state import OrchestratorState


# ============================================================
# 节点1: 分类
# ============================================================

def node_classify(_state: OrchestratorState) -> dict:
    """
    分类节点：调用三级分类器对来电文本进行分类。
    此函数由 Orchestrator 注入 classifier 实例。
    """
    # 从全局上下文获取分类器（在 orchestrator 中注入）
    return {}  # 实际由 orchestrator._node_classify 处理


# ============================================================
# 节点2: 画像查询
# ============================================================

def node_lookup_profile(_state: OrchestratorState) -> dict:
    """画像查询节点"""
    return {}


# ============================================================
# 节点3: 习惯推断
# ============================================================

def node_infer_presence(_state: OrchestratorState) -> dict:
    """习惯推断节点"""
    return {}


# ============================================================
# 节点4: 路由
# ============================================================

def node_route(_state: OrchestratorState) -> dict:
    """路由节点"""
    return {}


def route_by_type(state: OrchestratorState) -> str:
    """
    条件路由：根据 type_id 选择处理节点。

    返回: "scam" | "business" | "urgent" | "normal"
    """
    type_id = state.get("type_id", "general")

    if type_id in ("scam", "scam_risk", "telemarketing", "game_promo"):
        return "scam"
    if type_id in ("food_delivery", "express", "taxi_arrived", "bank"):
        return "business"
    if type_id in ("family", "leader", "urgent"):
        return "urgent"
    return "normal"


# ============================================================
# 节点5: 诈骗处理
# ============================================================

def node_handle_scam(_state: OrchestratorState) -> dict:
    """诈骗处理节点"""
    return {}


# ============================================================
# 节点6: 业务处理
# ============================================================

def node_handle_business(_state: OrchestratorState) -> dict:
    """业务处理节点（外卖/快递/银行等）"""
    return {}


# ============================================================
# 节点7: 紧急处理
# ============================================================

def node_handle_urgent(_state: OrchestratorState) -> dict:
    """紧急来电处理节点"""
    return {}


# ============================================================
# 节点8: 普通处理
# ============================================================

def node_handle_normal(_state: OrchestratorState) -> dict:
    """普通来电处理节点"""
    return {}


# ============================================================
# 节点9: 通知
# ============================================================

def node_notify(_state: OrchestratorState) -> dict:
    """通知节点"""
    return {}
