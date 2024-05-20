import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
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
    openai_api_key=os.getenv("OPENAI_KEY"),
    verbose=True
)

# Create the prompt TODO: This is used only for the "base chat" tool
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

# Initialize the chain TODO: Is used just for the "base chat" tool
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


def characterTool(query):
    values = query.split(",")
    characterUrl = values[0]
    characterName = values[1]

    req = ""
    if characterUrl:
        req = requests.get(characterUrl)
    elif characterName:
        req = requests.get(
            url="https://swapi.dev/api/people/",
            params={"search": characterName}
        )
    else:
        raise "Malformed query"

    if not req.status_code == 200:
        raise "Request error"
    response = req.text
    return response


# Create the search tools
tools = [
    # Tool.from_function(
    #     name="Star Wars Chat",
    #     description="For when you need to chat about Star Wars. The question will be a string. Return a string.",
    #     func=chat_chain.run,
    #     return_direct=True
    # ),
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
    ),
    Tool.from_function(
        name="Character Search",
        description="""
        Use when need to find details about characters given their url or name.

        Input:
        The input for this tool should be a comma separated list,
        where the first element is the character url, or empty character if not available,
        and the second element is the character name, or empty character if not available.
        Use this tool only if at least once of the tuple elements is available.
        Do not use this tool if none of the tuple elements are available.

        Output:
        This tool returns a stringified JSON with the following properties:
        "count": Total number of Characters matched. Characters are paginated
        "next": If not None, is the url to retrieve the next page of the results
        "previous": If not None, is the url to retrieve the previous page of the results. This value should be ignored
        "results": Is a stringified list of Characters.

        Each Character has the following attributes:

        name (of type string): The name of this person.
        birth_year (of type string): The birth year of the person, using the in-universe standard of BBY or ABY - Before the Battle of Yavin or After the Battle of Yavin. The Battle of Yavin is a battle that occurs at the end of Star Wars episode IV: A New Hope.
        eye_color (of type string): The eye color of this person. Will be "unknown" if not known or "n/a" if the person does not have an eye.
        gender (of type string): The gender of this person. Either "Male", "Female" or "unknown", "n/a" if the person does not have a gender.
        hair_color (of type string): The hair color of this person. Will be "unknown" if not known or "n/a" if the person does not have hair.
        height (of type string): The height of the person in centimeters.
        mass (of type string): The mass of the person in kilograms.
        skin_color (of type string): The skin color of this person.
        homeworld (of type string): The URL of a planet resource, a planet that this person was born on or inhabits.
        films (of type array): An array of film resource URLs that this person has been in.
        species (of type array): An array of species resource URLs that this person belongs to.
        starships (of type array): An array of starship resource URLs that this person has piloted.
        vehicles (of type array): An array of vehicle resource URLs that this person has piloted.
        url (of type string): the hypermedia URL of this resource.
        created (of type string): the ISO 8601 date format of the time that this resource was created.
        edited (of type string): the ISO 8601 date format of the time that this resource was edited.
        """,
        func=characterTool,
        return_direct=False
    )
]


# Then, pull from "Langchain hub" a pre-made agent
# This agent instructs the model to use the tools at disposal to answer the question
agent_prompt = hub.pull("hwchase17/react-chat")

agent = create_react_agent(llm, tools, prompt=agent_prompt)

# agent_prompt = hub.pull("hwchase17/openai-tools-agent")

# agent = create_tool_calling_agent(llm, tools, prompt=agent_prompt)

print("Agent prompt is: ")
agent_prompt.pretty_print()


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    max_interations=3,
    verbose=True,
    handle_parsing_errors=True
)


while True:
    q = input("> ")
    response = agent_executor.invoke({"input": q})
    print(response["output"])
