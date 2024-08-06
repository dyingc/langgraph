from abc import abstractmethod
import datetime
from json import tool
from unittest import result
from unittest.mock import Base
from typing import List, Annotated, cast, Optional, Literal
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, validator, Field
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, MessageGraph
from pydantic.v1.error_wrappers import ValidationError as ValidationError_Wrapper
from pydantic import SecretStr, ValidationError
from pydantic.v1.types import SecretStr as SecretStr_v1
from sqlalchemy import func
from utils import get_llm

class Reflection(BaseModel):
    """
    A focused reflection on key aspects of the writing process,
    including self-critique and suggestions for improvement.
    """

    content_quality: Annotated[
        str,
        Field(
            description="Assessment of the response's relevance, completeness, and conciseness. Identify key missing information and any superfluous content."
        ),
    ]

    accuracy_objectivity: Annotated[
        str,
        Field(
            description="Evaluation of the response's factual accuracy, potential biases, and any instances of unsupported claims or AI hallucinations."
        ),
    ]

    logical_strength: Annotated[
        str,
        Field(
            description="Analysis of the argument's logical consistency, the strength of supporting evidence, and any weak points in reasoning."
        ),
    ]


class AI_QAResponse(BaseModel):
    """
    A comprehensive representation of the AI-generated response to a user query,
    including the initial answer, self-reflection, and guidance for further research.
    This structure is designed to instruct LangGraph in performing subsequent online search work.
    """

    answer: Annotated[
        str,
        Field(
            description="The initial AI-generated response to the user query, typically in the form of a structured essay or detailed explanation."
        ),
    ]

    reflection: Annotated[
        Reflection,
        Field(
            description="A focused self-critique on key aspects of the response, identifying areas for improvement and potential issues."
        ),
    ]

    suggested_queries: Annotated[
        List[str],
        Field(
            description="A list of 3-5 specific search queries that could help address identified weaknesses and enhance the response's quality through further research.",
            min_items=3,
            max_items=5,
        ),
    ]

    confidence_score: Annotated[
        float,
        Field(
            description="A self-assessed score (0.0 to 1.0) indicating the AI's confidence in the accuracy and completeness of its initial response, guiding the need for additional research.",
            ge=0.0,
            le=1.0,
        ),
    ]

    priority_topics: Annotated[
        List[str],
        Field(
            description="A list of 2-3 key topics or areas from the response that would benefit most from additional research and verification.",
            min_items=2,
            max_items=3,
        ),
    ]

    @validator("confidence_score")
    def validate_confidence_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError(
                "The confidence score must be a value between 0.0 and 1.0."
            )
        return v

    @validator("priority_topics")
    def validate_priority_topics(cls, v):
        if len(v) < 2 or len(v) > 3:
            raise ValueError("The number of priority topics must be between 2 and 3.")
        return v

    @validator("suggested_queries")
    def validate_suggested_queries(cls, v):
        if len(v) < 3 or len(v) > 5:
            # if len(v) >=3 and len(v) <= 5:
            # raise ValidationError(errors=["The number of suggested queries must be between 3 and 5."], model=cls.__class__)
            raise ValueError("The number of suggested queries must be between 3 and 5.")
        return v


class AI_QARevision(AI_QAResponse):
    """
    A refined version of the AI-generated response, incorporating additional information and self-critique.
    This structure provides guidance on revising the initial answer to enhance its quality and accuracy.
    """

    references: Annotated[
        List[str],
        Field(
            description="A list of formatted references to be included in the revised answer, supporting key facts and claims."
        ),
    ]


# class EssayWritingState(BaseModel):
#     messages: List[BaseMessage] = Field(
#         List[BaseMessage],
#         description="All the messages that form the state of the MessageGraph",
#     )
#     iter_num: Annotated[int, Field(description="The number of iterations in the MessageGraph", default=0)] = 0
#     MAX_ITERATIONS: Annotated[int, Field(description="The maximum number of iterations in the MessageGraph", default=Literal[5])] = 5


class ReflectionUtils:

    draft_prompt_template = ChatPromptTemplate(
        name="draft_prompt",
        messages=[
            SystemMessage(
                content="""You are an expert researcher with exceptional analytical skills and a keen sense for thorough investigation. \
Your task is to respond to the given question with a well-structured, comprehensive essay. Follow these steps:
1. Analyze the question carefully, identifying key components and any implicit assumptions.
2. Outline the main points you plan to cover in your response, ensuring a logical flow of ideas.
3. Write a well-structured essay that includes:
    a) An engaging introduction that presents the topic and your main argument or thesis
    b) Body paragraphs that each focus on a distinct point, supported by evidence and examples
    c) A strong conclusion that summarizes your key points and provides a final insight
Throughout your essay, demonstrate:
    a) Critical thinking and analysis of multiple perspectives
    b) Integration of relevant facts, statistics, and expert opinions
    c) Consideration of potential counterarguments
    d) Clear and concise language appropriate for an academic audience

Current time: {current_time}

1. {first_instruction}
2. Provide your initial answer to the following query
3. Critically analyze your response, focusing on:
    a) Potential inaccuracies or gaps in information
    b) Logical inconsistencies or weak arguments
    c) Areas where the answer could be more comprehensive
    d) Possible biases or assumptions made
4. Based on your critique, suggest some specific search queries that could help address the identified weaknesses and improve the answer. Format each query as a bullet point."""
            ),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(
                content="""<reminder>Reflect on the users original question, the actions you've taken thus far and the materials provided by some external tools. Respond with function: {function_name}</reminder>"""
            ),
        ],
    )

    revise_instruction = """Revise your previous answer using the new information and self-reflection. Follow these guidelines carefully:

1. Content Enhancement:
   - Incorporate key information identified as missing in your previous critique.
   - Address any inaccuracies or biases highlighted in your reflection.
   - Strengthen weak arguments and improve logical consistency as per your earlier analysis.

2. Citation and Referencing:
   - Include numerical citations [1], [2], etc., throughout your revised answer to support key facts and claims.
   - Ensure every major point or statistic is backed by a citation.

3. Conciseness and Relevance:
   - Remove any superfluous information identified in your previous critique.
   - Focus on directly answering the original question with precision.
   - Aim for a word count around {word_num} words for the main answer body.

4. Structure and Clarity:
   - Organize your answer with a clear introduction, body, and conclusion.
   - Use topic sentences to improve the flow and readability of your response.

5. References Section:
   - Add a "References" section at the end of your answer (not counted in the word limit).
   - Format references as follows:
     [1] https://example.com
     [2] https://example.com
   - Ensure each citation in the text corresponds to a reference in this section.

6. Final Check:
   - Review your revised answer to ensure it addresses all aspects of the original question.
   - Verify that your answer falls around {word_num} word limit (excluding the References section).
   - Double-check that all citations are correctly formatted and correspond to the References section.

Remember, your goal is to provide a concise, well-supported, and accurate response that directly addresses the original query while incorporating the insights from your self-reflection and additional research."""

    def __init__(
        self,
        first_instruction: str,
        draft_validator: PydanticToolsParser,
        revise_validator: PydanticToolsParser,
        TAVILY_API_KEY: SecretStr_v1,
        try_num: int = 3,
        word_num: int = 500,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        if not llm:
            llm = cast(
                BaseChatModel,
                get_llm(
                    vendor="groq",
                    model="llama-3.1-70b-versatile",
                    temperature=0.5,
                    max_tokens=4096,
                ),
            )
            # llm = cast(BaseChatModel, get_llm(vendor="groq", model="llama-3.1-8b-instant", temperature=0.5, max_tokens=4096))
            llm.name = "Groq Researcher"
        current_time = lambda: datetime.datetime.now().isoformat()
        # self.draft_chain:RunnableSequence  = None
        self.draft_chain: RunnableSerializable = self.draft_prompt_template.partial(
            first_instruction=first_instruction,
            current_time=current_time,
            function_name=AI_QAResponse.__name__,
        ) | llm.bind_tools(tools=[AI_QAResponse])
        self.draft_chain.name = "Draft Chain"
        self.draft_validator = draft_validator

        self.revise_chain: RunnableSerializable = self.draft_prompt_template.partial(
            first_instruction=self.revise_instruction.format(word_num=word_num),
            current_time=current_time,
            function_name=AI_QARevision.__name__,
        ) | llm.bind_tools(tools=[AI_QARevision])
        self.revise_chain.name = "Revise Chain"
        self.revise_validator = revise_validator

        self.try_num = try_num
        self.tavily = TavilySearchResults(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY),
            max_results=3,
        )

    def draft_node(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Write the first draft of the essay and provide a self-reflection on the writing process.
        """
        for attempt in range(self.try_num):
            config: RunnableConfig = {"tags": [f"attempt:{attempt}"]}
            response = self.draft_chain.invoke(
                input={"messages": messages}, config=config
            )
            try:
                self.draft_validator.invoke(input=response)
                return response
            except ValidationError_Wrapper as e:
                messages.append(response)
                validation_message = ToolMessage(
                    content=f"Validation Error: {repr(e.errors())}\n\nRegenerating the response to fix the above validation error!",
                    tool_call_id=response.tool_calls[0]["id"],
                )
                messages.append(validation_message)
        raise ValueError(
            f"Failed to validate the response after {self.try_num} attempts!"
        )

    def run_quries(self, suggested_queries: List[str], **kwargs) -> List[str]:
        """
        Run the suggested queries and return the results.
        """
        results = self.tavily.batch(
            inputs=[{"query": query} for query in suggested_queries]
        )
        return results

    def research(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Doing researcch, based on the previous writing and reflection,
        including using tools to do online queries.
        """
        research_node = ToolNode(
            tools=[
                StructuredTool.from_function(func=self.run_quries, name=AI_QAResponse.__name__),
                StructuredTool.from_function(func=self.run_quries, name=AI_QARevision.__name__)
            ],
            name="Research Node"
        )
        results = research_node.invoke(input=messages)
        return results

    def revise(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Revise the answer based on the feedback from the reflection and research.
        """
        results:List[BaseMessage] = []
        revise_message = HumanMessage(content=self.revise_instruction)
        messages.append(revise_message)
        results.append(revise_message)
        for attempt in range(self.try_num):
            config: RunnableConfig = {"tags": [f"attempt:{attempt}"]}
            response = self.revise_chain.invoke(
                input={"messages": messages}, config=config
            )
            try:
                self.revise_validator.invoke(input=response)
                results.append(response)
                return results
            except ValidationError_Wrapper as e:
                # No need to append response or validation_message to the final results list if the validation fails
                messages.append(response)
                validation_message = ToolMessage(
                    content=f"Validation Error: {repr(e.errors())}\n\nRegenerating the response to fix the above validation error!",
                    tool_call_id=response.tool_calls[0]["id"],
                )
                messages.append(validation_message)
        raise ValueError(f"Failed to validate the response after {self.try_num} attempts!")

    def finish_revise(self, messages: List[BaseMessage]) -> Literal["__end__", "research"]:
        """
        Decide if we don't need more revise.
        We check the type of Messages in the `messages`, from back to the front.
        If we have more than four messages that are either not `ai` or `tool`, we stop the revise.
        @return:
            A string that indicates the next node in the graph.
        """
        count = 0
        thredshold = 4
        for m in reversed(messages):
            if m.type not in ["ai", "tool"]:
                count += 1
                if count >= thredshold:
                    return "__end__"
        return "research"


def main():
    from utils import get_llm, set_env, save_graph_image
    import os

    # Set the environment variables
    # os.environ['LANGCHAIN_API_KEY'] = ''
    set_env()
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph - Reflexion - 002"

    draft_validator = PydanticToolsParser(
        name="AIResponse Validator", tools=[AI_QAResponse]
    )
    revise_validator = PydanticToolsParser(
        name="AIResponse Validator", tools=[AI_QARevision]
    )
    reflection = ReflectionUtils(
        first_instruction="Write an essay on the following topic with about 300 words.",
        TAVILY_API_KEY=SecretStr_v1(os.environ["TAVILY_API_KEY"]),
        draft_validator=draft_validator,
        revise_validator=revise_validator,
        llm=cast(BaseChatModel, get_llm(vendor="openai", model="gpt-4o-mini", temperature=0.5, max_tokens=4096)),
    )
    messages: List[BaseMessage] = [
        HumanMessage(content="What is cryptocurrency and how does it work?")
    ]
    init_response = reflection.draft_node(messages=messages)
    print(init_response)
    messages.append(init_response)
    research_results = reflection.research(messages=messages)
    print(research_results)
    messages.extend(research_results)
    revise_results = reflection.revise(messages=messages)
    print(revise_results)
    messages.extend(revise_results)
    next = reflection.finish_revise(messages=messages)
    print(next)


if __name__ == "__main__":
    main()
