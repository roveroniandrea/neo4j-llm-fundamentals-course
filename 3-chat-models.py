from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

load_dotenv()

# Chats models are designed to receive in input a list of messages and return a chat-like response
chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)

# System messages provide intructions to the model
instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")

# Human messages provide questions (or any message) from a human
question = HumanMessage(content="What is the weather like?")

# Pass all the messages to the chat model
# System messages are usually passed as first
response = chat_llm.invoke([
    instructions,
    question
])

print(response.content)


# REFACTORING AS A CHAIN
prompt = PromptTemplate(
    template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Context: {context}
Question: {question}
""",
    input_variables=["question"],
)

chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

# Some context can be provided in the same message, so the chat model can answer accordingly
# In this case, the context is hardcoded, but it can be retrieved from an API
current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""


# The prompt seems to be treated as a System message, but notice that needs to pass the question in itself and the context
response = chat_chain.invoke({
    "context": current_weather,
    "question": "What is the weather like?"
})

print(response.get("text"))

# Let's ask for a more sophisticated answer
response = chat_chain.invoke({
    "context": current_weather,
    "question": "I'm looking for an easy beach to try surfing, which place do you recommend me?"
})

print(response.get("text"))
