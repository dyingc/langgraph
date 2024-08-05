import build
from utils import get_llm, set_env, save_graph_image
import os, json
from reflexion_utils import ReflectionUtils, AI_QAResponse
from langgraph.graph import END, MessageGraph, START
from langchain_core.output_parsers import PydanticToolsParser

# Set the environment variables
# os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - Reflexion'

system_task="Write an essay on the following topic with about 300 words."
output_validator = PydanticToolsParser(name="AIResponse Validator", tools=[AI_QAResponse])
reflection = ReflectionUtils(first_instruction=system_task, type_validator=output_validator)

builder = MessageGraph()
builder.add_node("draft", reflection.draft_node)
builder.add_node("research", reflection.research)
builder.add_node("revise", reflection.revise)

builder.add_edge(START, "draft")
builder.add_edge("draft", "research")
builder.add_edge("research", "revise")
builder.add_conditional_edges(source="revise",
                              path=reflection.finish_revise,
                              path_map={True: END, False: "research"})

graph = builder.compile()
save_graph_image(graph, "img/reflexion_002.png")

pass