import os

from dotenv import load_dotenv
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# Initialize the graph connection
graph = Neo4jGraph(
    url=os.getenv("NEO_4J_URL"),
    username=os.getenv("NEO_4J_USER"),
    password=os.getenv("NEO_4J_PW")
)

# Create the prompt template that will be used to generate Cypher queries
# This takes the database schema as input and the question for which a cypher query needs to be generated
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

# Create a chain that writes cypher queries
# I suppose that this automatically populates the prompt variables
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

# Ask questions, like "What role did Tom Hanks play in Toy Story?"
# Or ask for aggregate data, like "How many movies did Tom Hanks play?"

# Note that results are inconsistent and might not produce correct results
# For example, given the question "What role did Tom Hanks play in Toy Story?" twice, two different queries are generated:
#
# MATCH (a:Actor {name: "Tom Hanks"})-[:ACTED_IN]->(m:Movie {title: "Toy Story"})
# RETURN a.name, m.title, m.year, m.plot, m.poster, m.runtime, m.imdbRating
#
# Which works and returns "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."
#
# And the following:
# MATCH (a:Actor {name: "Tom Hanks"})-[:ACTED_IN {role: "Woody"}]->(m:Movie {title: "Toy Story"})
# RETURN a.name, m.title, "Woody" as role
#
# Which does not work because of the condition on the relationship
#
# Don't know exactly from where it comes the {role: "Woody"} condition, since this suggests that the LLM already knows the answer,
# But for now the inconsistent results are expected and will be fixed in the next lesson
#
# Fun fact: it might even write wrong queries:
# "Generated Cypher Statement is not valid"
# (even if it seems some error whyle parsing the query, because the generated text was this, and contains a valid query:
# 'To answer the question "What role did Tom Hanks play in Toy Story?", you can use the following Cypher query:\n\nMATCH (a:Actor {name: "Tom Hanks"})-[:ACTED_IN]->(m:Movie {title: "Toy Story"})\nRETURN a.name, m.title, m.year, m.plot, m.poster, m.imdbRating, m.revenue, m.budget, m.runtime, m.countries, m.languages, m.released, m.imdbVotes;'
# )
#
# In addition, this breaks if I ask for something not translatable to a query, like
# "What did I just ask?"
while True:
    q = input("> ")
    response = cypher_chain.invoke({"query": q})
    print(response["result"])
