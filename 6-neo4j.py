import os
from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph

load_dotenv()

# Neo4jGraph from langchain is a wrapper to the neo4j package
graph = Neo4jGraph(
    url=os.getenv("NEO_4J_URL"),
    username=os.getenv("NEO_4J_USER"),
    password=os.getenv("NEO_4J_PW")
)

result = graph.query("""
MATCH (m:Movie{title: 'Toy Story'}) 
RETURN m.title, m.plot, m.poster
""")

print(result)

print(graph.schema)
