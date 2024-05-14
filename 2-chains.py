from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import StrOutputParser
from langchain_openai import OpenAI
import os

load_dotenv()

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_KEY"),
    model="gpt-3.5-turbo-instruct",
    temperature=0
)


# METHOD 2: Chain consisting of prompt template and llm
# Chains allow to combine language models with different data sources and third party apis
# The simplest chain is an LLMChain. An LLMChain combines a prompt template with an LLM and returns a response
template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""")

llm_chain = LLMChain(
    llm=llm,
    prompt=template,
    # The StrOutputParser ensures that the response is a string
    # Optionally, a JSON structure can be specified in the prompt template, and a `SimpleJsonOutputParser` can be used to ensure the output is a valid JSON
    output_parser=StrOutputParser()
)

# The invoke method accepts template variables as dictinary
# The response will be a dictionary of all template variables and another property "text" with the answer
response = llm_chain.invoke({"fruit": "apple"})

print(response)
