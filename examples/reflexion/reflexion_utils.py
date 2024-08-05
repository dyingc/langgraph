from abc import abstractmethod
import datetime
from json import tool
from langchain_core.pydantic_v1 import BaseModel, validator, Field
from langchain_core.output_parsers import PydanticToolsParser
from typing import List, TypedDict, Annotated, cast
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic.v1.error_wrappers import ValidationError as ValidationError_Wrapper
from pydantic import ValidationError
from utils import get_llm

class Reflection(BaseModel):
    """
    A focused reflection on key aspects of the writing process, 
    including self-critique and suggestions for improvement.
    """

    content_quality: Annotated[str, Field(
        description="Assessment of the response's relevance, completeness, and conciseness. Identify key missing information and any superfluous content."
    )]

    accuracy_objectivity: Annotated[str, Field(
        description="Evaluation of the response's factual accuracy, potential biases, and any instances of unsupported claims or AI hallucinations."
    )]

    logical_strength: Annotated[str, Field(
        description="Analysis of the argument's logical consistency, the strength of supporting evidence, and any weak points in reasoning."
    )]

class AI_QAResponse(BaseModel):
    """
    A comprehensive representation of the AI-generated response to a user query,
    including the initial answer, self-reflection, and guidance for further research.
    This structure is designed to instruct LangGraph in performing subsequent online search work.
    """

    answer: Annotated[str, Field(
        description="The initial AI-generated response to the user query, typically in the form of a structured essay or detailed explanation."
    )]

    reflection: Annotated[Reflection, Field(
        description="A focused self-critique on key aspects of the response, identifying areas for improvement and potential issues."
    )]

    suggested_queries: Annotated[List[str], Field(
        description="A list of 3-5 specific search queries that could help address identified weaknesses and enhance the response's quality through further research.",
        min_items=3,
        max_items=5
    )]

    confidence_score: Annotated[float, Field(
        description="A self-assessed score (0.0 to 1.0) indicating the AI's confidence in the accuracy and completeness of its initial response, guiding the need for additional research.",
        ge=0.0,
        le=1.0
    )]

    priority_topics: Annotated[List[str], Field(
        description="A list of 2-3 key topics or areas from the response that would benefit most from additional research and verification.",
        min_items=2,
        max_items=3
    )]

    @validator("confidence_score")
    def validate_confidence_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("The confidence score must be a value between 0.0 and 1.0.")
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


class EssayWritingState(BaseModel):
    messages: List[BaseMessage] = Field(List[BaseMessage], description="All the messages that form the state of the MessageGraph")

class ReflectionUtils:
    draft_prompt_template = ChatPromptTemplate(name="draft_prompt", messages=[
        SystemMessage(content="""You are an expert researcher with exceptional analytical skills and a keen sense for thorough investigation. \
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
4. Based on your critique, suggest 1~3 specific search queries that could help address the identified weaknesses and improve the answer. Format each query as a bullet point."""),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="""<reminder>Reflect on the users original question, the actions you've taken thus far and the materials provided by some external tools. Respond with function: {function_name}</reminder>"""),
    ])

    @abstractmethod
    def __init__(self, first_instruction:str, type_validator:PydanticToolsParser, try_num:int=3) -> None:
        llm = get_llm(vendor="groq", model="llama-3.1-70b-versatile", temperature=0.5, max_tokens=4096)
        # llm = get_llm(vendor="groq", model="llama-3.1-8b-instant", temperature=0.5, max_tokens=4096)
        llm = cast(BaseChatModel, llm)
        llm.name = "Groq Researcher"
        current_time = lambda: datetime.datetime.now().isoformat()
        # self.draft_chain:RunnableSequence  = None
        self.draft_chain:RunnableSerializable = self.draft_prompt_template.partial(first_instruction=first_instruction, current_time=current_time , function_name=AI_QAResponse.__name__) | llm.bind_tools(tools=[AI_QAResponse])
        self.draft_chain.name = "Draft Chain"
        self.type_validator = type_validator
        self.try_num = try_num

    @abstractmethod
    def draft_node(self, state:EssayWritingState)->AIMessage:
        """
        Write the first draft of the essay and provide a self-reflection on the writing process.
        """
        for attempt in range(self.try_num):
            config:RunnableConfig = {"tags": [f"attempt:{attempt}"]}
            response = self.draft_chain.invoke(input={"messages": state.messages}, config=config)
            try:
                self.type_validator.invoke(input=response)
                return response
            except ValidationError_Wrapper as e:
                state.messages.append(response)
                validation_message = ToolMessage(content=f"Validation Error: {repr(e.errors())}\n\nRegenerating the response to fix the above validation error!",
                                                 tool_call_id=response.tool_calls[0]['id'])
                state.messages.append(validation_message)

    @abstractmethod
    def research(self, state:EssayWritingState):
        """
        Doing researcch, based on the previous writing and reflection, 
        including using tools to do online queries.
        """
        pass

    @abstractmethod
    def revise(self, state:EssayWritingState):
        pass

    @abstractmethod
    def finish_revise(self, state:EssayWritingState)->bool:
        """
        Decide if we don't need more revise.
        @return: 
            True if the essay is ready to be submitted
            False if the essay needs more revision
        """
        pass

def main():
    from utils import get_llm, set_env, save_graph_image
    import os

    # Set the environment variables
    # os.environ['LANGCHAIN_API_KEY'] = ''
    set_env()
    os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - Reflexion - 002'

    output_validator = PydanticToolsParser(name="AIResponse Validator", tools=[AI_QAResponse])
    reflection = ReflectionUtils(first_instruction="Write an essay on the following topic with about 300 words.", type_validator=output_validator)
    messages:List[BaseMessage]=[HumanMessage(content="What is cryptocurrency and how does it work?")]
    response = reflection.draft_node(EssayWritingState(messages=messages))
    print(response)

if __name__ == "__main__":
    main()