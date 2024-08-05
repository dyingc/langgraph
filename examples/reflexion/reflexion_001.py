# Import necessary modules and functions
from utils import get_llm, set_env, save_graph_image
import os

# Set the environment variables
# os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - Reflexion'

# Import Tavily search tools and utilities
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.pydantic_v1 import SecretStr

# Initialize Tavily search API wrapper and tool
search = TavilySearchAPIWrapper(tavily_api_key=SecretStr(os.environ['TAVILY_API_KEY']))
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

# Import necessary modules for message handling, output parsing, and prompts
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableSequence
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError, validator
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Annotated, cast

# Get the language model and cast it to BaseChatModel
llm = get_llm(vendor="groq", model="llama-3.1-70b-versatile", temperature=0.5, max_tokens=2048)
# llm = get_llm(vendor="groq", model="llama-3.1-8b-instant", temperature=0.5, max_tokens=2048)
# llm = get_llm(vendor="openai", model="gpt-4o-mini", temperature=0.5, max_tokens=2048)
llm = cast(BaseChatModel, llm)

# Define the Reflection model with fields for missing and superfluous critiques
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous or unnecessary.")
    inconsistent_logic: str = Field(description="Critique of inconsistent logic in the answer.")
    hallucination: str = Field(description="Critique of hallucination - providing information that is not true, in the answer.")

# Define the AnswerQuestion model with fields for answer, reflection, and search queries
class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""
    answer: str = Field(description="Around 250 word (more than 200 and less than 300) detailed answer to the question.")
    # answer: Annotated[str, Field(description="~250 word detailed answer to the question.")]
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")

    # @validator("answer")
    # def check_answer_length(cls, v):
    #     if len(v.split()) > 300 or len(v.split()) < 200:
    #         raise ValidationError(errors=["The answer should be around 250 words."], model=AnswerQuestion)
    #     return v
    
    # @validator("search_queries")
    # def check_search_queries_length(cls, v):
    #     if len(v) > 3 or len(v) < 1:
    #         raise ValidationError(errors=["You should provide 1-3 search queries."], model=AnswerQuestion)

# Define a class to handle responses with retries
class ResponderWithRetries:
    def __init__(self, runnable:RunnableSequence, validator:PydanticToolsParser):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: list)->AIMessage:
        response = []
        for attempt in range(3):
            config:RunnableConfig = {"tags": [f"attempt:{attempt}"]}
            response = self.runnable.invoke(input={"messages": state}, config=config)
            try:
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        raise RuntimeError("Failed to respond after 3 attempts.")

# Import datetime module
import datetime

# Define a chat prompt template for the actor
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<reminder>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# Define the initial answer chain with the prompt template and language model
initial_answer_chain:RunnableSequence = cast(RunnableSequence, actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion]))

# Define a validator for the initial answer
validator = PydanticToolsParser(tools=[AnswerQuestion])

# Create a responder with retries for the initial answer
first_responder = ResponderWithRetries(runnable=initial_answer_chain, validator=validator)

# Example question to be answered
example_question = "Why nuclear weapon is not an answer to the world's global peace?"
# initial = first_responder.respond([HumanMessage(content=example_question)])
initial = AIMessage(content="")

# Instructions for revising the answer
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is between 200 ~ 300 words.
"""

# Extend the initial answer schema to include references
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,
    cite your reflection with references, and finally
    add search queries to improve the answer."""
    references: list[str] = Field(description="Citations motivating your updated answer.")

# Define the revision chain with the prompt template and language model
revision_chain:RunnableSequence = cast(RunnableSequence, actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer]))

# Define a validator for the revised answer
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

# Create a responder with retries for the revised answer
revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

# Import json module
import json

# Generate the revised answer
pass
if len(initial.tool_calls) > 0:
    query_result = [ToolMessage(
                    tool_call_id=initial.tool_calls[0]["id"],
                    content=json.dumps(
                        tavily_tool.invoke(
                            input={"query": q}, max_results=3
                        )
                    ),
                    ) for q in initial.tool_calls[0]['args']['search_queries']]
else:
    query_result = [ToolMessage(content="", tool_call_id='')]

# revised = revisor.respond(
#     [
#         HumanMessage(content=example_question),
#         initial,
#         *query_result,
#     ]
# )
# print(revised)

# Import necessary modules for structured tools and tool nodes
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

# Define a function to run the generated queries
def run_queries(search_queries: list[str], **kwargs):
    """
    Run the generated queries.
    Args:
        search_queries (list[str]): The search queries to run. This is defined in the AnswerQuestion model.
    """
    return tavily_tool.batch([{"query": query} for query in search_queries])

# Create a tool node with structured tools for running queries
tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

# Import necessary modules for message graph and typing
from typing import Literal, cast
from langgraph.graph import END, MessageGraph, START

# Define the maximum number of iterations
MAX_ITERATIONS = 5

# Create a message graph builder
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

from reflexion_utils import ReflectionUtils

# Define edges between nodes in the graph
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# Define a function to get the number of iterations
def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i

# Define the event loop for the graph
def event_loop(state: list) -> Literal["execute_tools", "__end__"]:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

# Add conditional edges to the graph
builder.add_conditional_edges("revise", event_loop)
builder.add_edge(START, "draft")

# Compile the graph
graph = builder.compile()
save_graph_image(graph, "img/reflexion_exercise_001.png")
pass