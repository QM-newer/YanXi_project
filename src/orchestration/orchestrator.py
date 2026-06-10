"""
主调度器 (Orchestrator)
========================
基于 LangGraph StateGraph 的多智能体调度系统。

从 CC 项目的 CallOrchestrator 整合而来，
集成双路混合检索（BM25 + 向量语义）。

流程:
  STT文本 → classify → lookup_profile → infer_presence → route → action → notify
"""

from langgraph.graph import StateGraph, END

from src.core.enums import CallType, CallAction, PresenceMode
from src.core.llm_client import LLMClient
from src.classification.classifier import ThreeTierClassifier
from src.orchestration.state import OrchestratorState
from src.orchestration.nodes import route_by_type
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CallOrchestrator:
    """
    基于 LangGraph 的多智能体调度器。

    完整流程:
      来电文本 → 分类 → 画像查询 → 习惯推断 → 路由 → 动作 → 通知
    """

    def __init__(self, config: dict):
        self.config = config

        # --- LLM 客户端 ---
        self.llm_client = LLMClient(config)

        # --- 嵌入模型 ---
        self.embedder = None
        try:
            self.embedder = Embedder(config)
        except Exception as e:
            logger.warning(f"嵌入模型初始化失败: {e}")

        # --- 双路混合检索（BM25 + 向量） ---
        self.hybrid_retriever = HybridRetriever(config, self.embedder)

        # --- 分类器 ---
        self.classifier = ThreeTierClassifier(
            config,
            retriever=self.hybrid_retriever,
            llm_client=self.llm_client,
        )

        # --- 重排序器 ---
        hybrid_cfg = config.get("hybrid_retrieval", {})
        self.reranker = Reranker(
            embedder=self.embedder,
            semantic_weight=hybrid_cfg.get("reranker_semantic_weight", 0.5),
            keyword_weight=hybrid_cfg.get("reranker_keyword_weight", 0.3),
        )

        # --- 机主状态 ---
        self.presence_mode = PresenceMode.FREE.mode

        # --- 构建 LangGraph ---
        self.graph = self._build_graph()
        logger.info("CallOrchestrator 初始化完成（BM25+向量双路检索）")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        双路混合检索（统一检索入口）。

        参数:
            query: 查询文本
            top_k: 返回数量

        返回:
            list: 融合后的文档列表
        """
        # 双路检索 + RRF 融合
        fused = self.hybrid_retriever.retrieve(query, top_k=top_k)

        # 语义重排序
        if self.reranker and fused:
            fused = self.reranker.rerank(query, fused, top_n=top_k)

        return fused[:top_k]

    def index_documents(self, documents: list) -> None:
        """构建检索索引"""
        self.hybrid_retriever.index_documents(documents)

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        graph = StateGraph(OrchestratorState)

        # 添加节点
        graph.add_node("classify", self._node_classify)
        graph.add_node("lookup_profile", self._node_lookup_profile)
        graph.add_node("infer_presence", self._node_infer_presence)
        graph.add_node("route", self._node_route)
        graph.add_node("handle_scam", self._node_handle_scam)
        graph.add_node("handle_business", self._node_handle_business)
        graph.add_node("handle_urgent", self._node_handle_urgent)
        graph.add_node("handle_normal", self._node_handle_normal)
        graph.add_node("notify", self._node_notify)

        # 设置入口
        graph.set_entry_point("classify")

        # 边
        graph.add_edge("classify", "lookup_profile")
        graph.add_edge("lookup_profile", "infer_presence")
        graph.add_edge("infer_presence", "route")
        graph.add_conditional_edges(
            "route",
            route_by_type,
            {
                "scam": "handle_scam",
                "business": "handle_business",
                "urgent": "handle_urgent",
                "normal": "handle_normal",
            },
        )
        graph.add_edge("handle_scam", "notify")
        graph.add_edge("handle_business", "notify")
        graph.add_edge("handle_urgent", "notify")
        graph.add_edge("handle_normal", "notify")
        graph.add_edge("notify", END)

        return graph.compile()

    # ============================================================
    # LangGraph 节点
    # ============================================================

    def _node_classify(self, state: OrchestratorState) -> dict:  # type: ignore[no-untyped-def]
        """分类节点"""
        text = state.get("call_text", "")
        if not text:
            return {
                "type_id": "general",
                "call_type_name": "未知",
                "confidence": 0.0,
                "classify_method": "empty",
            }

        result = self.classifier.classify(text)
        return {
            "type_id": result.type_id,
            "call_type_name": result.call_type.display_name,
            "confidence": result.confidence,
            "classify_method": result.method,
        }

    def _node_lookup_profile(self, state: OrchestratorState) -> dict:  # type: ignore[no-untyped-def, arg-type]
        """画像查询节点"""
        return {"caller_profile": None}

    def _node_infer_presence(self, state: OrchestratorState) -> dict:  # type: ignore[no-untyped-def, arg-type]
        """习惯推断节点"""
        return {"presence_mode": self.presence_mode, "presence_reason": ""}

    def _node_route(self, state: OrchestratorState) -> dict:
        """路由节点"""
        type_id = state.get("type_id", "general")
        presence_mode = state.get("presence_mode", "free")

        # 诈骗类 → 拒接
        if type_id in ("scam", "scam_risk", "telemarketing", "game_promo"):
            return {"final_action": CallAction.REJECT.value}

        # 重要联系人类 → 转接
        if type_id in ("family", "leader", "urgent"):
            return {"final_action": CallAction.FORWARD.value}

        # 业务类 → 代接
        if type_id in ("food_delivery", "express", "taxi_arrived", "bank"):
            return {"final_action": CallAction.PROXY.value}

        # 普通来电
        if presence_mode in ("busy", "dnd"):
            return {"final_action": CallAction.PROXY.value}
        return {"final_action": CallAction.GENERAL_REPLY.value}

    def _node_handle_scam(self, state: OrchestratorState) -> dict:
        """诈骗处理节点"""
        text = state.get("call_text", "")
        confidence = state.get("confidence", 0.8)

        # 检索相关案例
        try:
            retrieval_docs = self.retrieve(text, top_k=3)
            _context = "\n".join([d.get("content", "")[:200] for d in retrieval_docs])
        except Exception:
            _context = ""

        # LLM 分析
        try:
            analysis = self.llm_client.chat(
                messages=[{
                    "role": "user",
                    "content": f"分析以下来电是否为诈骗，简要说明理由：\n{text}"
                }],
                default_response="疑似诈骗电话",
            )
        except Exception:
            analysis = "疑似诈骗电话"

        # 构建通知卡片
        card = {
            "title": "🚫 诈骗拦截",
            "body": f"类型: {state.get('call_type_name', '诈骗')}\n{analysis[:100]}",
            "priority": "high",
            "type": "scam",
            "confidence": confidence,
        }

        return {
            "agent_reply": "",
            "notification_card": card,
            "final_action": CallAction.REJECT.value,
        }

    def _node_handle_business(self, state: OrchestratorState) -> dict:
        """业务处理节点（外卖/快递/银行等）"""
        text = state.get("call_text", "")
        type_id = state.get("type_id", "general")

        # 根据类型生成回复
        reply_map = {
            "food_delivery": "好的，请放在门口，谢谢！",
            "express": "好的，麻烦放菜鸟驿站，谢谢！",
            "taxi_arrived": "好的，我马上下来。",
            "bank": "好的，我知道了，谢谢通知。",
        }
        reply = reply_map.get(type_id, "好的，我知道了。")

        # 构建通知卡片
        type_name = state.get("call_type_name", "业务")
        card = {
            "title": f"📦 {type_name}通知",
            "body": text[:100],
            "priority": "normal",
            "type": type_id,
        }

        return {
            "agent_reply": reply,
            "notification_card": card,
            "final_action": CallAction.SUMMARY_CARD.value,
        }

    def _node_handle_urgent(self, state: OrchestratorState) -> dict:
        """紧急来电处理节点"""
        text = state.get("call_text", "")
        type_name = state.get("call_type_name", "紧急")

        reply = "好的，我马上通知机主。"

        card = {
            "title": f"⚠️ {type_name}",
            "body": text[:100],
            "priority": "high",
            "type": "urgent",
        }

        return {
            "agent_reply": reply,
            "notification_card": card,
            "final_action": CallAction.FORWARD.value,
        }

    def _node_handle_normal(self, state: OrchestratorState) -> dict:
        """普通来电处理节点"""
        text = state.get("call_text", "")
        _presence_mode = state.get("presence_mode", "free")
        final_action = state.get("final_action", CallAction.GENERAL_REPLY.value)

        if final_action == CallAction.PROXY.value:
            reply = "您好，机主现在不方便接听电话。请问您有什么事，我可以帮忙转达。"
            card = {
                "title": "📞 代接来电",
                "body": text[:100],
                "priority": "normal",
                "type": "normal",
            }
        else:
            reply = "好的，我知道了。"
            card = None

        return {
            "agent_reply": reply,
            "notification_card": card,
            "final_action": final_action,
        }

    def _node_notify(self, state: OrchestratorState) -> dict:
        """通知节点"""
        card = state.get("notification_card")
        if card:
            logger.info(f"通知卡片: {card.get('title', '')}")
        return {}

    # ============================================================
    # 公共接口
    # ============================================================

    def run(self, call_text: str, caller_number: str = "") -> dict:
        """
        处理一次来电。

        参数:
            call_text: STT 识别的来电文本
            caller_number: 来电号码

        返回:
            dict: 处理结果
        """
        initial_state = {
            "call_text": call_text,
            "caller_number": caller_number,
        }

        try:
            result = self.graph.invoke(initial_state)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}")
            result = {
                "final_action": "error",
                "final_message": str(e),
                "agent_reply": "",
            }

        return dict(result)

    def resume_conversation(self, _previous_result: dict, new_text: str) -> dict:
        """
        继续多轮对话。

        参数:
            _previous_result: 上一轮 run() 的返回值（保留参数兼容）
            new_text: 来电者新说的话

        返回:
            dict: 处理结果
        """
        logger.info(f"继续对话: {new_text[:50]}...")

        # 重新分类
        classify_result = self.classifier.classify(new_text)
        new_type = classify_result.type_id

        # 如果新分类变成诈骗 → 立即拒接
        if new_type in ("scam", "scam_risk", "telemarketing", "game_promo"):
            card = {
                "title": "🚫 诈骗拦截（多轮检测）",
                "body": f"检测到诈骗特征: {new_text[:100]}",
                "priority": "high",
                "type": "scam",
                "confidence": classify_result.confidence,
            }
            return {
                "final_action": CallAction.REJECT.value,
                "type_id": new_type,
                "call_type_name": classify_result.call_type.display_name,
                "confidence": classify_result.confidence,
                "classify_method": classify_result.method,
                "agent_reply": "",
                "notification_card": card,
            }

        # 业务类继续对话
        if new_type in ("food_delivery", "express", "taxi_arrived", "bank"):
            reply_map = {
                "food_delivery": "好的，谢谢！",
                "express": "好的，谢谢！",
                "taxi_arrived": "好的，马上到！",
                "bank": "好的，我知道了。",
            }
            reply = reply_map.get(new_type, "好的，我知道了。")
            return {
                "final_action": CallAction.SUMMARY_CARD.value,
                "type_id": new_type,
                "call_type_name": classify_result.call_type.display_name,
                "confidence": classify_result.confidence,
                "classify_method": classify_result.method,
                "agent_reply": reply,
            }

        # 其他类型：礼貌结束
        return {
            "final_action": CallAction.GENERAL_REPLY.value,
            "type_id": new_type,
            "call_type_name": classify_result.call_type.display_name,
            "confidence": classify_result.confidence,
            "classify_method": classify_result.method,
            "agent_reply": "好的，我知道了。请问还有其他事吗？",
        }

    def set_presence(self, mode: str, reason: str = "") -> None:
        """设置机主状态"""
        self.presence_mode = mode
        logger.info(f"机主状态: {mode} ({reason})")
