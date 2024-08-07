import build
from utils import get_llm, set_env, save_graph_image
import os, json
from reflexion_utils import ReflectionUtils, AI_QAResponse, AI_QARevision
from langgraph.graph import END, MessageGraph, START
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic.v1.types import SecretStr as SecretStr_v1
from typing import cast

# Set the environment variables
# os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - Reflexion - 002'

system_task="Write an essay on the following topic with about 500 words."
draft_validator = PydanticToolsParser(name="AIResponse Validator", tools=[AI_QAResponse])
revise_validator = PydanticToolsParser(name="AIResponse Validator", tools=[AI_QARevision])
llm = get_llm(vendor="openai", model="gpt-4o-mini", temperature=0.5, max_tokens=4096)
# llm = get_llm(vendor="groq", model="llama-3.1-70b-versatile", temperature=0.5, max_tokens=4096)
# llm = get_llm(vendor="groq", model="llama-3.1-8b-instant", temperature=0.5, max_tokens=4096)
reflection = ReflectionUtils(first_instruction=system_task,
                             draft_validator=draft_validator,
                             revise_validator=revise_validator,
                             llm=cast(BaseChatModel, llm),
                             TAVILY_API_KEY=SecretStr_v1(os.environ['TAVILY_API_KEY']))

builder = MessageGraph()
builder.add_node("draft", reflection.draft_node)
builder.add_node("research", reflection.research)
builder.add_node("revise", reflection.revise)

builder.add_edge("draft", "research")
builder.add_edge("research", "revise")
builder.add_conditional_edges(source="revise",
                              path=reflection.finish_revise,
                              path_map={END: END, "research": "research"})

builder.set_entry_point("draft")
graph = builder.compile()
save_graph_image(graph, "img/reflexion_002.png")

messages = [HumanMessage(content="What is LangGraph and how it differs with Crew AI?")]
graph.invoke(input=messages)

pass