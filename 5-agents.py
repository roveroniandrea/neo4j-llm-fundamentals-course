from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
import os
from langchain_community.tools import YouTubeSearchTool

# Agents wrap a LLM and gives it a set of "tools" that the model can access to retrieve data
# Example of tools are APIs, data sources (I think dbs) or functionality (file system access maybe?)
# The agent itself chooses which tool to use to fullfill a task


load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# See prev lessons for infos
prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Create the youtube search tool
youtube = YouTubeSearchTool()

# Create a tool to use for chats regarding movies
tools = [
    Tool.from_function(
        # Name and description allows the LLM to choose which tool to use for each need
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        # This specifies which function to call when the tool is invoked
        # So, this tool just asks the model itself for an answer, without accessing any API
        func=chat_chain.run,
        # Don't actually know what it does: "The return_direct flag indicates that the tool will return the result directly"
        # UPDATE: Found it by trying in file #12: If True, it means that the output of the tool will be directedly shown as the answer
        # If False, the output of the tool will be passed and elaborated by the LLM
        return_direct=True,
    ),
    # This tool instead, allows to seach for a trailer on youtube and retrieve a link
    Tool.from_function(
        name="Movie Trailer Search",
        # The description tells to use it when the question includes the "trailer" keyword
        # So the LLM should not pick it for other questions
        description="Use when needing to find a movie trailer. The question will include the word 'trailer'. Return a link to a YouTube video.",
        func=youtube.run,
        return_direct=True
    )

]

# Then, pull from "Langchain hub" a pre-made agent
# This agent instructs the model to use the tools at disposal to answer the question
agent_prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(
    # The agent uses the previously initialized chat model
    llm,
    # The agent uses the tools
    tools,
    # And starts from the pre-made agent
    agent_prompt
)

# I think the Agent executor wrap the agent allowing to chat with him
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    # This prevents the model from running too long or entering an infinite loop
    max_interations=3,
    # Used for debugging, prints on the console the tool execution, like which tool has been picked, which inout is passed, which output returns
    verbose=True,
    # If true, handles parsing errors in case they occur and returns a message to the user
    handle_parsing_errors=True
)


# Trying with different prompt:
# When asking "Tell me the plot of the movie "The Searchers"", the model uses the Movie Chat tool
# Instead, when asking "Show me a trailer of that movie" the model uses the Movie Trailer Search with Action input ""The Searchers" trailer"

# I also note that the resulting links are the two first links found when using an incognito window and searching
# ""The Searchers" trailer" on Youtube navigation bar
while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
