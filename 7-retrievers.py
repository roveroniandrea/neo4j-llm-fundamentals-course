import os

from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# NOTE: This lesson does not refer to Labradors

# Retrievers are chains that allow to retrieve unstructured documents (aka text) given an unstructured query
# Example of a query might be "Find a movie plot about a robot that wants to be human."

# I'm not entirely sure what's the difference between retrievers and tools then
# (UPDATE): I think this is a tool in fact. In next chapter I add this as a tool that the agent can use

load_dotenv()

# Retrievers often use a vector store (ie vector index) to search for data
# Neo4jVector is a vector store that can generate embeddings and store/retrieve them on Neo4j

# The embedding provider is used to generate a vector embedding of each query (and I suppose of each data if needed)
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_KEY")
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=os.getenv("NEO_4J_URL"),
    username=os.getenv("NEO_4J_USER"),
    password=os.getenv("NEO_4J_PW"),
    # This is the name of the index created on a past lesson online
    index_name="moviePlots",
    # This is the node property that contains the embedding. Like m.embedding where m is a Movie node
    embedding_node_property="embedding",
    # This is the node property that contains the text (ie the movie plot)
    # Don't know why this is necessary, since the embedding should be sufficient.
    # I suppose this is used to generate an embedding when creating or updating a node
    # Also, this is used to populate "doc.page_content" on each result
    text_node_property="plot",
)

# Search for similarity with the query
# What I suppose it happens is that an embedding of the query is generated
# then the embedding is seached for similarities on the vector index above
result = movie_plot_vector.similarity_search(
    "A movie where aliens land and attack earth.",
    # Number of documents to retrieve
    k=4
)

for doc in result:
    # page_contens is the value of the "text_node_property"
    # metadata in the Movie node
    print(doc.metadata["title"], "-", doc.page_content)


# Now that we have a vector store, we can create a retriever chain (or retrieval chain)
# This extends a vector store, allowing to use it inside a Langchain application

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# The RetrievalQA class is a chain that uses a retriever as part of its pipeline
plot_retriever = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=movie_plot_vector.as_retriever(),
    # These two flags allow for better understanding what's happening, by verbosing the output and populating "source_documents" with matched documents
    verbose=True,
    return_source_documents=True
)

# What happens is that the vector store is first used to match movies against the query
# Then the result (as Documents) is passed to the LLM to elaborate the response
# In fact, the documents matched are 4, (see result.source_documents), but the LLM chooses the first one since I asked "A (single) movie"
# NOTE: This is not entirely right. If I search "Find me a movie where a young guy travers with a scientist called "Doc" with a time machine"
# that apparently are not on the vector store, the result["result"] correctly says Return to the Future, but source_documents contain some other movies
result = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(result)
