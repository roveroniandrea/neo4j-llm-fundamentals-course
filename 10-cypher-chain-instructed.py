import os

from dotenv import load_dotenv
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

graph = Neo4jGraph(
    url=os.getenv("NEO_4J_URL"),
    username=os.getenv("NEO_4J_USER"),
    password=os.getenv("NEO_4J_PW")
)

# The previous lesson we saw the LLM could provide inexact or not valid queries
# Add an instruction to the prompt to tell the LLM to only use relationships and properties already present in the schema
# This should prevent the LLM to invent new properties or conditions
# Instructions might be expressed both on positive (do this) and negative (don't do this) ways
#
# Another instruction gives more context on the stored data, for example in movies starting with "The", "The" is placed at the end
#
# Other instructions tell to respond only to questions that can generate cypher queries, and to do not try to answer when the query returns nothing
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".
If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.
Do not include any text except the generated Cypher statement.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

# Still not really good.
# When I ask "What role did Tom Hanks play in Toy Story?" sometimes he tries to use a condition on the relationship,
# with an undefined query variable "role" and it breaks
#
# Also, when asked for a question not regarding the graph ("Hello, how are you?")
# He enters the chain with an invalid query "I'm here to help with Neo4j queries. What is your question related to movies?" and it breaks
while True:
    q = input("> ")
    response = cypher_chain.invoke({"query": q})
    print(response["result"])
