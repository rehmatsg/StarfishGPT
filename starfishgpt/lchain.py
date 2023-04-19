from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from typing import Optional


# Set up a prompt template
class _CustomPromptTemplate(BaseChatPromptTemplate):
  # The template to use
  template: str
  # The list of tools available
  tools: List[Tool]

  def format_messages(self, **kwargs) -> str:
    # Get the intermediate steps (AgentAction, Observation tuples)
    # Format them in a particular way
    intermediate_steps = kwargs.pop("intermediate_steps")
    thoughts = ""
    for action, observation in intermediate_steps:
      thoughts += action.log
      thoughts += f"\nObservation: {observation}\nThought: "
    # Set the agent_scratchpad variable to that value
    kwargs["agent_scratchpad"] = thoughts
    # Create a tools variable from the list of tools provided
    kwargs["tools"] = "\n".join(
      [f"{tool.name}: {tool.description}" for tool in self.tools])
    # Create a list of tool names for the tools provided
    kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
    formatted = self.template.format(**kwargs)
    return [HumanMessage(content=formatted)]


class _CustomOutputParser(AgentOutputParser):

  def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
    # Check if agent should finish
    if "Final Answer:" in llm_output:
      return AgentFinish(
        # Return values is generally always a dictionary with a single `output` key
        # It is not recommended to try anything else at the moment :)
        return_values={
          "output": llm_output.split("Final Answer:")[-1].strip()
        },
        log=llm_output,
      )
    # Parse out the action and action input
    regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    match = re.search(regex, llm_output, re.DOTALL)
    if not match:
      raise ValueError(f"Could not parse LLM output: `{llm_output}`")
    action = match.group(1).strip()
    action_input = match.group(2)
    # Return the action and action input
    return AgentAction(tool=action,
                       tool_input=action_input.strip(" ").strip('"'),
                       log=llm_output)


_llm = ChatOpenAI(temperature=0)
_output_parser = _CustomOutputParser()


class AIFunctions:

  @staticmethod
  def chat(system_prompt: str, prompt: str, tools: Optional = []):
    template = system_prompt + """
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Tool: the action to take, should be one of [{tool_names}]
Tool Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {prompt}
{agent_scratchpad}
"""
    langchain_prompt = _CustomPromptTemplate(
      template=template,
      tools=tools,
      # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
      # This includes the `intermediate_steps` variable because that is needed
      input_variables=["prompt", "intermediate_steps"])
    llm_chain = LLMChain(llm=_llm, prompt=langchain_prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(llm_chain=llm_chain,
                                 output_parser=_output_parser,
                                 stop=["\nObservation:"],
                                 allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True)
    agent_executor.run(prompt)
    return