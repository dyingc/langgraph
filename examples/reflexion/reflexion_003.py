from abc import abstractmethod
from utils import get_llm, set_env, save_graph_image
import os, json
from reflexion_utils import ReflectionUtils, AI_QAResponse, AI_QARevision
from reflexion_utils_003 import ReflexAgentState
from langgraph.graph import START, END, StateGraph
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pydantic.v1.types import SecretStr as SecretStr_v1
from typing import Annotated, cast, Literal

# Set the environment variables
os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - Reflexion - 003'

builder = StateGraph(state_schema=ReflexAgentState)

@abstractmethod
def plan(state: ReflexAgentState):
    pass

@abstractmethod
def research(state: ReflexAgentState):
    pass

@abstractmethod
def compose(state: ReflexAgentState):
    pass

@abstractmethod
def self_critique(state: ReflexAgentState)->Literal['NEXT', 'REDO']:
    return 'NEXT'

@abstractmethod
def critique(state: ReflexAgentState):
    pass

@abstractmethod
def revise(state: ReflexAgentState):
    pass

@abstractmethod
def post_revise(state: ReflexAgentState)->Literal['REDO', 'PUBLISH', 'PLAN', 'RESEARCH', 'COMPOSE']:
    return 'PUBLISH'

@abstractmethod
def publish(state: ReflexAgentState):
    pass
    
builder.add_node("planner", plan)
builder.add_node("rechercher", research)
builder.add_node("composer", compose)
builder.add_node("critiquer", critique)
builder.add_node("reviser", revise)
builder.add_node("publisher", publish)

builder.add_edge(START, "planner")
builder.add_conditional_edges(source="planner", path=self_critique,
                              path_map= {'NEXT': "rechercher", 'REDO': "planner"})
builder.add_conditional_edges(source="rechercher", path=self_critique,
                              path_map= {'NEXT': "composer", 'REDO': "rechercher"})
builder.add_conditional_edges(source="composer", path=self_critique,
                              path_map= {'NEXT': "critiquer", 'REDO': "composer"})
builder.add_edge("critiquer", "reviser")
builder.add_conditional_edges(source="reviser", path=post_revise,
                              path_map= {'REDO': "reviser",
                                         'PUBLISH': "publisher",
                                         'PLAN': "planner",
                                         'RESEARCH': "rechercher",
                                         'COMPOSE': "composer"})

builder.add_edge("publisher", END)


graph = builder.compile()

save_graph_image(graph=graph, dest_png_path="img/reflexion_003.png")