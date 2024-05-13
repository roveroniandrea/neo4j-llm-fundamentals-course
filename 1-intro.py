from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser
from langchain_openai import OpenAI
import os

load_dotenv()

# Use OpenAI as the model. Langchain supports other models like Cohere and Ollama

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_KEY"),
    model="gpt-3.5-turbo-instruct",
    # Low temperature means less random results, thus more grounded to fatcs.
    # On the opposite, temperature of 1 means more creative answers but increased possibility of hallucinations
    temperature=0 
)

# This allows to pass a custom prompt
# response = llm.invoke("What is Neo4j?")

# While these allows to use the prompt template
# Promt templates allow to create reausable questions and give more context to the LLM
template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm.invoke(template.format(fruit="apple"))

print(response)