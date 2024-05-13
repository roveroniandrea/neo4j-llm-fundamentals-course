from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

load_dotenv()

chat_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY")
)


# A memory allos a chat model to keep track of the conversation.
# Without it, the conversation might go in circles, for example the model asking every time the user's name

# Since every invocation of the chat model is stateless, the memory must be provided in every call
prompt = PromptTemplate(template="""You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.

Chat History: {chat_history}
Context: {context}
Question: {question}
""", input_variables=["chat_history", "context", "question"])

# There are different memory types depending on the case, but in this tutorial the `ConversationBufferMemory` is used
memory = ConversationBufferMemory(
    # This is the prompt variable that needs to be populated with the chat history
    memory_key="chat_history",
    # This is the prompt variable that is populated with the user's question. This will be added to the memory
    input_key="question",
    # If true, specifies that the conversation history will be returned as a list
    # In fact, I can double-check this by inspecting response["chat_history"] variable,
    # that shows some message list (not including the prompt)
    return_messages=True
)

chat_chain = LLMChain(
    llm=chat_llm,
    prompt=prompt,
    # The memory is passed as an argument
    memory=memory,
    # Used for debugging, shows the conversation history in the console
    verbose=False
)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

response = chat_chain.invoke({
    "context": current_weather,
    "question": "Hi, I am at Watergate Bay. What is the surf like?"
})

print(response["text"])

response = chat_chain.invoke({
    "context": current_weather,
    "question": "Where I am?"
})

# It should answer "at Watergate Bay"
print(response["text"])


# Experimenting with a loop
while True:
    question = input("> ")
    response = chat_chain.invoke({
        "context": current_weather,
        "question": question
        })

    print(response["text"])