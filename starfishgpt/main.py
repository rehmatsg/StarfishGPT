# from starfishgpt.tools.search import search

# results = search("AutoGPT")

# print(results)

import yaml
import os
# from starfishgpt.openai import OAI, OModel
from starfishgpt.toolbox import ToolboxAI
from starfishgpt.task_master import TaskMaster
import openai

with open("config.yaml", mode="rt", encoding="utf-8") as file:
  config = yaml.safe_load(file)

openai.api_key = os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# from starfishgpt.lchain import AIFunctions

# system_prompt = '''
# You are Elon Musk, CEO of Tesla and SpaceX. You are being interviewed. Since you are weird, you always reverse your message.
# '''

# prompt = 'Tell me something about yourself'

# result = AIFunctions.chat(
#   system_prompt=system_prompt,
#   prompt=prompt,
#   tools=[reverse_string]
# )

# result = OAI.call(
#   model=OModel.GPT3,
#   prompt=prompt,
#   system_prompt=system_prompt
# )

# print(result)

# system_prompt = "You have been designed to help your user with any of his requests."

# prompt = "What is your name?"

# ToolboxAI.call(
#   prompt="['Search AutoGPT on Google', 'Summarise what AutoGPT does', 'Explain in simple words']"
# )

master = TaskMaster.create(
  prompt="Create a python function to search for wikipedia articles and return the main content of the first result"
)
print(master)
output = master.run("google")
print(type(output))
print(output)