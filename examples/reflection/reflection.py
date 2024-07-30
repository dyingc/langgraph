from utils import set_env, get_llm
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

# Set the environment
os.environ['LANGCHAIN_API_KEY'] = ''
set_env()
os.environ['LANGCHAIN_PROJECT'] = 'LangGraph - reflection'

prompt_msg = """You are an essay assistant tasked with writing excellent 5-paragraph essays. \
Generate the best essay possible for the user's request. \
If the user provides critique, respond with a revised version of your previous attempts.

Topic: {topic}"""

prompt: PromptTemplate = PromptTemplate(name="Essay Assistant", template=prompt_msg)
llm = get_llm(vendor='groq', model='llama-3.1-70b-versatile', max_tokens=2048, temperature=0.7)
chain = prompt | llm

for chunk in chain.stream(input={'topic': 'the importance of education in society'}):
    msg = chunk.content
    id = chunk.id
    print(msg, end='')