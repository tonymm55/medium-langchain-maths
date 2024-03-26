from langchain_openai import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
  name="Wikipedia",
  func=wikipedia.run,
	description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")

problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
  name="Calculator",
  func=problem_chain.run,
  description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")

word_problem_template = """You are a reasoning agent tasked with solving 
the user's logic-based questions. Logically arrive at the solution, and be 
factual. In your answers, clearly detail the steps involved and give the 
final answer. Provide the response in bullet points. 
Question  {question} Answer"""

math_assistant_prompt = PromptTemplate(
  input_variables=["question"],
  template=word_problem_template)

word_problem_chain = LLMChain(
  llm=llm,
  prompt=math_assistant_prompt)

word_problem_tool = Tool.from_function(
  name="Reasoning Tool",
  func=word_problem_chain.run,
  description="Useful for when you need to answer logic-based/reasoning questions.",
)

agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, word_problem_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

print(agent.invoke({"input": "I have 3 apples and 4 oranges. I give half of my oranges away and buy two dozen new ones, alongwith three packs of strawberries. Each pack of strawberry has 30 strawberries. How  many total pieces of fruit do I have at the end?"}))