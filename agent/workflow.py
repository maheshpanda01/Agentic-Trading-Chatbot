from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *

# Number of messages before summarization kicks in
SUMMARIZE_AFTER = 10

class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str  # stores the running summary of old messages

class GraphBuilder:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.tools = [retriever_tool, financials_tool, tavilytool]
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.graph = None

    def _summarize_node(self, state: BotState):
        """
        Summarizes old messages when conversation gets too long.
        Keeps only the last 2 messages + a running summary.
        """
        messages = state["messages"]
        summary = state.get("summary", "")

        # Only summarize if we exceed the threshold
        if len(messages) <= SUMMARIZE_AFTER:
            return {}

        # Build summarization prompt
        if summary:
            summary_prompt = (
                f"This is the summary of the conversation so far:\n{summary}\n\n"
                f"Extend the summary by including the new messages above. "
                f"Keep it concise but preserve all important trading information, "
                f"stock names, financial figures, and key decisions discussed."
            )
        else:
            summary_prompt = (
                "Summarize the conversation above in a concise paragraph. "
                "Preserve all important trading information, stock names, "
                "financial figures, and key decisions discussed."
            )

        # Messages to summarize = all except last 2
        messages_to_summarize = messages[:-2]
        recent_messages = messages[-2:]

        # Ask LLM to summarize
        summarization_input = messages_to_summarize + [HumanMessage(content=summary_prompt)]
        new_summary = self.llm.invoke(summarization_input).content

        # Remove old messages from state (keep only last 2)
        messages_to_delete = [RemoveMessage(id=m.id) for m in messages_to_summarize]

        return {
            "summary": new_summary,
            "messages": messages_to_delete
        }

    def _chatbot_node(self, state: BotState):
        """
        Main chatbot node. Prepends summary as context if it exists.
        """
        summary = state.get("summary", "")
        messages = state["messages"]

        # Build system message — include summary if exists
        system_content = """You are a trading research assistant.
        You have access to these tools:
        - retriever_tool: ALWAYS use this first when the user asks to summarize, analyze,
          explain, or find information from uploaded documents, files, reports, or the knowledge base.
        - tavilytool: Use for real-time web search, market news, and current financial trends.
        - financials_tool: Use for stock financial statements using ticker symbols like AAPL, TSLA, MSFT.
        Always remember previous messages in the conversation and maintain context."""

        if summary:
            system_content += f"\n\nSummary of earlier conversation:\n{summary}"

        system = SystemMessage(content=system_content)
        return {"messages": [self.llm_with_tools.invoke([system] + messages)]}

    def build(self):
        graph_builder = StateGraph(BotState)

        # Add nodes
        graph_builder.add_node("summarize", self._summarize_node)
        graph_builder.add_node("chatnode", self._chatbot_node)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        # Add edges
        graph_builder.add_edge(START, "summarize")       # always check summarization first
        graph_builder.add_edge("summarize", "chatnode")  # then go to chatbot
        graph_builder.add_conditional_edges("chatnode", tools_condition)
        graph_builder.add_edge("tools", "chatnode")

        conn = sqlite3.connect("memory.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        self.graph = graph_builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph is not built. Call build() first.")
        return self.graph
