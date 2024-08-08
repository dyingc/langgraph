from abc import abstractmethod
import datetime
import uuid
import copy
from json import tool
from unittest import result
from unittest.mock import Base
from typing import Any, List, Annotated, cast, Optional, Literal, Dict
from enum import Enum
from click import Option
from groq import APIError, BadRequestError
from langchain_core.pydantic_v1 import BaseModel, validator, Field
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables.base import RunnableSerializable, RunnableBinding
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, MessageGraph
from more_itertools import first
from pydantic.v1.error_wrappers import ValidationError as ValidationError_Wrapper
from pydantic import SecretStr, ValidationError
from pydantic.v1.types import SecretStr as SecretStr_v1

from sqlalchemy import func
from sympy import fu
from utils import get_llm


class Plan(BaseModel):
    class _SubTask(BaseModel):
        task: Annotated[str, Field(description="The sub-task to be completed.")]
        description: Annotated[
            str,
            Field(
                description="A brief description of the sub-task.",
                title="Description",
            ),
        ]
    
    # task: Annotated[str, Field(description="The original text of the task.")]

    objective: Annotated[
        str,
        Field(
            description="Describe the main goal or desired outcome of the task.",
            title="Objective",
        ),
    ]
    subtasks: Annotated[
        Optional[List[_SubTask]],
        Field(
            description="The subtasks that need to be completed to achieve the main goal. No need to generate sub-tasks if the major one is already clear.",
            title="Subtasks", min_items=0, max_items=3
        ),
    ]

class ReflexAgentState(BaseModel):
    task : Annotated[str, "The task to be completed"]
    plan: Annotated[Optional[Plan], "The plan of the task, given by the planner"] = None


class Planner():
    prompt_template : PromptTemplate = PromptTemplate.from_template("""
                                                                  Generate a plan for the following task: {task}

Provide the plan in this format:
1. Objective: Describe the main goal in one sentence.
2. Subtasks (up to 3, if needed):
   - Task: [Brief task description]
   - Description: [Short explanation of the subtask]

Only include subtasks if the main task needs to be broken down.
Respond with function: {function_name}""")

    def __init__(self, llm: BaseChatModel, validator: PydanticToolsParser, retry_limit:int=3):

        self.llm : BaseChatModel = llm
        self.llm.name = "Planner"
        self.validator : PydanticToolsParser = validator
        self.validator.name = "Planner Schema Validator"

        self.planning_chain: RunnableSerializable = (
            self.prompt_template.partial(
                function_name=", ".join([t.__name__ for t in validator.tools])
            )
            | self.llm.bind_tools(validator.tools) # Ensure the output format complies with the schema
        )
        self.planning_chain.name = "Planner Chain"
        self.retry_limit = retry_limit

    def do_plan(self, agent_state:ReflexAgentState)->List[Plan]:
        for i in range(self.retry_limit):
            try:
                response = self.planning_chain.invoke({'task': agent_state.task})
            except Exception as e:
                pass
            # Check if the response is valid for the designated schema
            try:
                result: List[Plan] = self.validator.invoke(response)
                return result
            except ValidationError as e:
                pass
        response = self.planning_chain.invoke(agent_state.task)
        return response

def test():
    from utils import get_llm, set_env, save_graph_image
    import os

    # Set the environment variables
    # os.environ['LANGCHAIN_API_KEY'] = ''
    set_env()
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph - Reflexion Utils - 003"
    llm = get_llm()
    llm = get_llm(vendor="groq", model="llama-3.1-70b-versatile", temperature=0.5, max_tokens=4096)
    llm = get_llm(vendor="groq", model="llama-3.1-8b-instant", temperature=0.5, max_tokens=4096)
    llm = cast(BaseChatModel, llm)
    planner_validator = PydanticToolsParser(name="Planner Validator", strict=True, tools=[Plan])
    planner = Planner(llm=llm, validator=planner_validator, retry_limit=3)
    plans = planner.do_plan(ReflexAgentState(task="Write an essay about quantum physics"))
    print(plans)


if __name__ == "__main__":
    test()
