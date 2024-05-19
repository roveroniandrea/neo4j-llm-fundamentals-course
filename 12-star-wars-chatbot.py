import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import YouTubeSearchTool
from langchain_openai import ChatOpenAI
import requests

# This is a challenge to create a chatbot about anything I want
# I choose to create a LLM to which I can chat about Star Wars
# This LLM uses a custom tool that uses SWAPI REST API to retrieve informations
# It also has conversational memory


load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# Create the prompt
prompt = PromptTemplate(
    template="""You are a Star Wars chatbot.
    You answer questions regarding Star Wars films and characters, species, planets, starships and vehicles that appear in Star Wars.

    Answer only questions regarding Star Wars.
    For questions that do not regard Star Wars, answer "I can only answer about Star Wars questions"
    
    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"]
)

# Conversational memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the chain
chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)


def filmTool(query):
    values = query.split(",")
    movieUrl = values[0]
    movieTitle = values[1]

    req = ""
    if movieUrl:
        req = requests.get(movieUrl)
    elif movieTitle:
        req = requests.get(
            url="https://swapi.dev/api/films/",
            params={"search": movieTitle}
        )
    else:
        raise "Malformed query"

    if not req.status_code == 200:
        raise "Request error"

    response = req.text

    return response


# Create the search tools
tools = [
    Tool.from_function(
        name="Film Search",
        description="""
        Use when need to find movie details given their title or url.

        Input:
        The input for this tool should be a comma separated list,
        where the first element is the movie url, or empty character if not available,
        and the second element is the movie title, or empty character if not available.
        Use this tool only if at least once of the tuple elements is available.
        Do not use this tool if none of the tuple elements are available.

        Output:
        This tool returns a stringified JSON with the following properties:
        "count": Total number of Films matched. Films are paginated
        "next": If not None, is the url to retrieve the next page of the results
        "previous": If not None, is the url to retrieve the previous page of the results. This value should be ignored
        "results": Is a stringified list of Films.

        Each film has the following attributes:

        title (of type string): The title of this film
        episode_id (of type integer): The episode number of this film.
        opening_crawl (of type string): The opening paragraphs at the beginning of this film.
        director (of type string): The name of the director of this film.
        producer (of type string): The name(s) of the producer(s) of this film. Comma separated.
        release_date (of type date) : The ISO 8601 date format of film release at original creator country.
        species (of type array): An array of species resource URLs that are in this film.
        starships (of type array): An array of starship resource URLs that are in this film.
        vehicles (of type array): An array of vehicle resource URLs that are in this film.
        characters (of type array): An array of people resource URLs that are in this film.
        planets (of type array): An array of planet resource URLs that are in this film.
        url (of type string): the hypermedia URL of this resource.
        created (of type string): the ISO 8601 date format of the time that this resource was created.
        edited (of type string): the ISO 8601 date format of the time that this resource was edited.
        """,
        func=filmTool,
        return_direct=False
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

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_interations=3,
    verbose=True,
    handle_parse_errors=True
)


while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
