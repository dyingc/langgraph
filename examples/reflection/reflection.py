from utils import set_env, get_llm
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

# Set the environment
os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - reflection'

essay_prompt_msg = """You are an essay assistant tasked with writing excellent 5-paragraph essays. \
Generate the best essay possible for the user's request. \
If the user provides critique, respond with a revised version of your previous attempts.

<essay_topic>
{topic}
</essay_topic>"""

essay_prompt: PromptTemplate = PromptTemplate(name="Essay Writing prompt", template=essay_prompt_msg)
student = get_llm(vendor='groq', model='llama-3.1-70b-versatile', max_tokens=2048, temperature=1.0)
essay_writing = essay_prompt | student
essay_writing.name = "Essay Writing"


essay_topic = "Why neuclear weapons can't resolve conflicts?"
count = 0
essay = ''
# for chunk in essay_writing.stream(input={'topic': essay_topic}):
#     msg = chunk.content
#     id = chunk.id
#     essay += msg
#     print(msg, end='')
#     count+=1
# print(f"\n\nTotal student's chunks returned: {count:,d}\n\n\n\n")

critic_prompt_msg = """You are a teacher tasked with providing feedback on a student's essay. \
You need to generate critique and recommendations for the student's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc. \

<requested_essay_topic>
{essay_topic}
</requested_essay_topic>

<essay>: 
{essay}
</essay>"""

critic_prompt: PromptTemplate = PromptTemplate(name="Essay Critic prompt", template=critic_prompt_msg)
teacher = get_llm(vendor='groq', model='llama-3.1-70b-versatile', max_tokens=512, temperature=0.2)

essay_critique = critic_prompt | teacher
essay_critique.name = "Essay Critique"

# count = 0
# critique = ''
# for chunk in essay_critique.stream(input={"essay_topic": essay_topic, "essay": essay}, config={"configurable": {"thread_id": 1}}):
#     msg = chunk.content
#     critique += msg
#     id = chunk.id
#     print(msg, end='')
#     count+=1
# print(f"\n\nTotal critique chunks returned: {count:,d}\n\n\n")

refine_prompt_msg = """You are an essay assistant tasked with writing excellent essays. \
Generate the best essay possible for the user's request. \
If the user provides critique, respond with a revised version of your previous attempts. \
Revised version only without mention why you made the changes or the list of the modifications.

<essay_topic>
{topic}
</essay_topic>

<previous_attempts>
{essay}
</previous_attempts>

<critique>
{critique}
</critique>"""

refine_prompt: PromptTemplate = PromptTemplate(name="Essay Refinement prompt", template=refine_prompt_msg)
essay_refiner = refine_prompt | student
essay_refiner.name = "Essay Refinement"

# count = 0
# refined_essay = ''
# for chunk in essay_refiner.stream(input={"topic": essay_topic, "essay": essay, "critique": critique}):
#     msg = chunk.content
#     id = chunk.id
#     refined_essay += msg
#     print(msg, end='')
#     count+=1

translater = get_llm(vendor='groq', model='llama-3.1-70b-versatile', max_tokens=4096, temperature=0.2)
translation_prompt_msg = """You are a language assistant tasked with translating the essay into another language. \
Translate the following texts (essay, essay's critique and the refined essay) into Simplified Chinese. Put your \
output into the corresponding sections.

Essay:
<essay>
{essay}
</essay>

Critique:
<critique>
{critique}
</critique>

Refined Essay:
<refined_essay>
{refined_essay}
</refined_essay>
"""
translation_prompt = PromptTemplate(name="Translation prompt", template=translation_prompt_msg)
translating = translation_prompt | translater
translating.name = "Translation"

# count = 0
# translation = ''
# for chunk in translating.stream(input={"essay": essay, "critique": critique, "refined_essay": refined_essay}):
#     msg = chunk.content
#     translation += msg
#     id = chunk.id
#     print(msg, end='')
#     count+=1
# print(f"\n\nTotal translation chunks returned: {count:,d}\n\n\n")


# Using LangGraph

from langgraph.graph import StateGraph

# Define Agent State
from typing import TypedDict, Annotated
class EssayAgentState(TypedDict):
    task: Annotated[str, "The description of the task, e.g. 'Write an essay about why nuclear weapons can't resolve conflicts'"] = ''
    essay: Annotated[str, "The essay written by the student"] = ''
    critique: Annotated[str, "The critique provided by the teacher"] = ''
    iterate: Annotated[int, "The number of iterations to refine the essay"] = 0

# Define a coroutine function to generate the essay
async def write_essay_node(state: EssayAgentState) -> EssayAgentState:
    if state.get('critique') is not None and state.get('essay') is not None:
        content = await essay_refiner.ainvoke({'topic': state['task'], 'essay': state['essay'], 'critique': state['critique']})
    else:
        content = await essay_writing.ainvoke({'topic': state['task']})
    essay = content['content']
    return {**state, 'essay': essay}

async def test(essay_topic: str):
    await write_essay_node({'task': essay_topic})

from asyncio import run, gather, create_task
if __name__ == '__main__':
    run(test("Why neuclear weapons can't resolve conflicts?"))
