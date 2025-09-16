from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from app.config import OPTIMIZER_MODEL
from app.prompt import PromptOptimizerPrompt


async def build_agent():
    llm = ChatOpenAI(model_name=OPTIMIZER_MODEL, temperature=0.7)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                PromptOptimizerPrompt.system_prompt
            ),
            HumanMessagePromptTemplate.from_template(PromptOptimizerPrompt.user_prompt),
        ]
    )
    return llm, prompt


async def optimize_prompt(user_prompt: str):
    llm, prompt = await build_agent()
    chain = prompt | llm

    inputs = {"user_prompt": user_prompt}

    result = await chain.ainvoke(inputs)
    raw_output = str(result.content)
    return raw_output
