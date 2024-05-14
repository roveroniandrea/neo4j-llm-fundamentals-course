import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import YouTubeSearchTool
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# Initialize the prompt template
prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

# Create the conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the chain
chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)


# Create the youtube search tool
youtube = YouTubeSearchTool()


# Also, create the Retriever
embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_KEY")
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=os.getenv("NEO_4J_URL"),
    username=os.getenv("NEO_4J_USER"),
    password=os.getenv("NEO_4J_PW"),
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    # These two flags allow for better understanding what's happening, by verbosing the output and populating "source_documents" with matched documents
    verbose=True,
    return_source_documents=True
)

# Tools expect a single query input and a single output key.
# Since RetrievalQA chain returns multiple outputs ("result" and "source_documents"), it needs to be wrapped


def run_retriever(query):
    results = plot_retriever.invoke({"query": query})
    # format the results
    # Why using results["source_documents"] and not results["result"] ?
    # In fact, what I see is that if the movies are not listed on the vectot store, source_documents could be some other movies (not too much related)
    # But "result" would be the right movie anyway (see ending comment)
    movies = '\n'.join([doc.metadata["title"] + " - " +
                       doc.page_content for doc in results["source_documents"]])
    return movies


# Create tools for the agens
tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=chat_chain.run,
        return_direct=True,
    ),
    # This tool searches on youtube
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True
    ),
    # This instead, uses the neo4j Retriever
    Tool.from_function(
        name="Movie search by plot",
        description="Use when needing to find one or more movie from a given plot. The question will be a string. Return a string.",
        func=run_retriever,
        return_direct=True
    )
]


# Then, pull from "Langchain hub" a pre-made agent
# This agent instructs the model to use the tools at disposal to answer the question
agent_prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(
    llm,
    tools,
    agent_prompt
)

# I think the Agent executor wrap the agent allowing to chat with him
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_interations=3,
    verbose=True,
    handle_parse_errors=True
)


# Now chat with the agent

# NOTE: If I search "Find me a movie where a young guy travers with a scientist called "Doc" with a time machine"
# that apparently are not on the vector store, the result["result"] correctly says Return to the Future, but source_documents contain some other movies
# So If I keep results["source_documents"] the LLM might answer with some garbage movies, despite knowing the solution
# So I'm not entirely sure what calls are made when choosing the tool, and what happens when the tools answers back
# My question is: if the tool does not find the right movie and returns garbage,
# why (and by who) the field results["result"] is pupolated with "I think you're referring to Return to the Future"
while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
