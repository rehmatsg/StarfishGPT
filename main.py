# from starfishgpt.tools.search import search

# results = search("AutoGPT")

# print(results)

from starfishgpt.lchain import AIFunctions

system_prompt = '''
You are Elon Musk, CEO of Tesla and SpaceX. You are currently in an interview with the user. Answer the questions as best as you can
'''

result = AIFunctions.chat(
  system_prompt=system_prompt,
  prompt="Tell me something about yourself"
)