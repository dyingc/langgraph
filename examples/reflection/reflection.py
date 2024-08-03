# https://langchain-ai.github.io/langgraph/cloud/deployment/setup/

from utils import set_env, get_llm, save_graph_image
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START
from langgraph.checkpoint import MemorySaver
from langgraph.graph.state import CompiledStateGraph

larger_model = "llama-3.1-70b-versatile"
smaller_model = "llama-3.1-8b-instant"
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
student = get_llm(vendor='groq', model=smaller_model, max_tokens=4096, temperature=0.7)
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
teacher = get_llm(vendor='groq', model=larger_model, max_tokens=2048, temperature=0.2)

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

# translater = get_llm(vendor='groq', model=smaller_model, max_tokens=4096, temperature=0.2)
translator = get_llm(vendor='openai', model="gpt-4o-mini", max_tokens=8192, temperature=0.2)
translation_prompt_msg = """You are a highly skilled language assistant specializing in translation. Your task is to translate the following essay-related content from its original language into {to}. Please maintain the tone, style, and nuances of the original text while ensuring the translation is natural and fluent in the target language.

Instructions:
1. Translate the original essay, its critique, and the refined essay.
2. Preserve the formatting and structure of each section.
3. Ensure that technical terms, idiomatic expressions, and cultural references are accurately conveyed.
4. Maintain the <essay>, <critique>, and <refined_essay> tags in your response.

Please provide your translations in the following format:

Translated Essay:
<essay>
[Your translation of the original essay goes here]
</essay>

Translated Critique:
<critique>
[Your translation of the critique goes here]
</critique>

Translated Refined Essay:
<refined_essay>
[Your translation of the refined essay goes here]
</refined_essay>

Original content for translation:

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
translating = translation_prompt | translator
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
    task: Annotated[str, "The specific essay prompt or writing assignment given to the student"]
    essay: Annotated[str, "The current version of the essay written by the student"]
    critique: Annotated[str, "Detailed feedback and suggestions provided by the teacher on the current essay"]
    prior_essay: Annotated[str, "The previous version of the essay, if any, used as a reference for improvements"]
    translated_essay: Annotated[str, "The final essay translated into a specified target language"]
    iterate: Annotated[int, "The current number of revision iterations completed"]
    max_iterate: Annotated[int, "The maximum number of revision iterations allowed for the essay"]

# Define a coroutine function to generate the essay
async def write_essay_node(state: EssayAgentState) -> EssayAgentState:
    new_state = EssayAgentState(**state)
    if state.get('critique') is not None and state.get('essay') is not None:
        content = await essay_refiner.ainvoke({'topic': state['task'], 'essay': state['essay'], 'critique': state['critique']})
        prior_essay = state['essay']
        new_state['prior_essay'] = prior_essay
    else:
        content = await essay_writing.ainvoke({'topic': state['task']})
    essay = content.content
    new_state['essay'] = essay
    new_state['iterate'] += 1
    return {**new_state}

async def critique_essay_node(state: EssayAgentState) -> EssayAgentState:
    content = await essay_critique.ainvoke({'essay_topic': state['task'], 'essay': state['essay']})
    critique = content.content
    return {**state, 'critique': critique}

async def translate_essay_node(state: EssayAgentState) -> EssayAgentState:
    content = await translating.ainvoke({'essay': state['prior_essay'], 'critique': state['critique'], 'refined_essay': state['essay'], 'to': 'Simplified Chinese'})
    translation = content.content
    return {**state, 'translated_essay': translation}


async def move_on(state: EssayAgentState) -> bool:
    return state['iterate'] < state['max_iterate']

async def get_assistant()->CompiledStateGraph:
    graph_builder = StateGraph(state_schema=EssayAgentState)
    graph_builder.add_node('write_essay', write_essay_node)
    graph_builder.add_node('critique_essay', critique_essay_node)
    graph_builder.add_node('translate_essay', translate_essay_node)
    graph_builder.add_edge(start_key=START, end_key='write_essay')
    graph_builder.add_conditional_edges(source='write_essay', path=move_on, path_map={True: 'critique_essay', False: 'translate_essay'})
    graph_builder.add_edge(start_key='critique_essay', end_key='write_essay')
    graph_builder.add_edge(start_key='translate_essay', end_key=END)
    graph = graph_builder.compile(checkpointer=MemorySaver())
    save_graph_image(graph=graph, dest_png_path="img/essay_agent_graph.png")
    return graph

async def test(essay_topic: str):
    graph :CompiledStateGraph = await get_assistant()
    initial_state = EssayAgentState(task=essay_topic, essay='', critique='', iterate=0, max_iterate=2)
    config = {"configurable": {"thread_id": 1}}
    # response = await graph.ainvoke(input={'task': essay_topic}, config={"configurable": {"thread_id": 1}})
    prior_output = {}
    async for event in graph.astream_events(input=initial_state, config=config, version='v2'):
        # print(event)
        # print(event['data'].get('output', ''))
        if event.get('event') == "on_chain_end":
            curr_state = await graph.aget_state(config)
            output = curr_state.values
            if prior_output != output:
                print(output)
                print("\n\n\n")
                prior_output = output
        #     for k, v in event['data']['output'].items():
        #         print(f"{k}:\n{v}")
        #     print("\n\n\n")
    pass

from asyncio import run, gather, create_task
if __name__ == '__main__':
    run(test("Why neuclear weapons can't resolve conflicts?"))
