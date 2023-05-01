from starfishgpt.openai import OAI, OModel
from typing import Optional

demo_tools = [
  {
    "name": "analyze_code",
    "description": "This function analyzes a given code string and returns suggestions for improvement. The function uses the OpenAI Codex model to generate suggestions based on the given code.",
    "use-cases": "This function is useful for programmers who want to improve the quality of their code or learn new coding techniques."
  },
  {
    "name": "execute_python_file",
    "description": "This function executes a Python file with the given filename.",
    "use-cases": "This function is useful for running Python scripts from within another Python program."
  },
  {
    "name": "execute_terminal_command",
    "description": "This function executes a terminal command",
    "use-cases": "This function is useful for executing terminal commands"
  },
  {
    "name": "append_to_file",
    "description": "This function appends the given text to the end of the specified file.",
    "use-cases": "This function is useful for writing to log files or other types of files that require ongoing updates."
  },
  {
    "name": "delete_file",
    "description": "This function deletes the specified file.",
    "use-cases": "This function is useful for deleting files that are no longer needed."
  },
  {
    "name": "read_file",
    "description": "This function reads the contents of the specified file and returns it as a string.",
    "use-cases": "This function is useful for reading data from files that are needed for further processing."
  },
  {
    "name": "search_files",
    "description": "This function searches for files in the specified directory and returns a list of file names that match the search criteria.",
    "use-cases": "This function is useful for finding files that match specific criteria, such as files with a certain extension or files that contain a particular string."
  },
  {
    "name": "write_to_file",
    "description": "This function writes the given text to the specified file, overwriting any existing content in the file.",
    "use-cases": "This function is useful for writing to files that need to be completely overwritten with new content."
  },
  {
    "name": "search",
    "description": "This function performs a Google search using the given query and returns top 5 results.",
    "use-cases": "This function is useful for finding information on the internet."
  },
  {
    "name": "scrape_webpage",
    "description": "This function scrapes the given webpage and find summarises the main contents of the page",
    "use-cases": "This function is useful for finding information on a webpage."
  },
  {
    "name": "generate_image",
    "description": "This function generates an image based on the given prompt using the OpenAI DALL-E model.",
    "use-cases": "This function is useful for generating images for use in various applications, such as graphic design or web development."
  },
  {
    "name": "improve_code",
    "description": "This function takes a code string and a list of suggestions and returns an improved version of the code string. The function uses the OpenAI Codex model to generate the improved code based on the given suggestions.",
    "use-cases": "This function is useful for programmers who want to improve the quality of their code or learn new coding techniques."
  },
  {
    "name": "browse_website",
    "description": "This function opens a web browser and navigates to the specified URL. If a question is provided, the function searches the page for the answer to the question.",
    "use-cases": "This function is useful for automated web browsing and data collection."
  },
  {
    "name": "summarise",
    "description": "This function is summarises the given text",
    "use-cases": "This function is useful for summarising long texts and keeping only relevant information"
  }
]

# assistant_reply_format = """
# You should only respond in the XML format as described below.
# Response Format:
# <Thoughts>...</Thoughts>
# <Reasoning>...</Reasoning>
# <Plan> short numbered list that convey long-term plans </Plan>
# <Criticism> constructive self criticism </Criticism>
# <Speak> summarise your thoughts, reasoning, plan and criticism to say to the user </Speak>
# <Tool name="..."/>

# You cannot repeat any of these tags more than 1 time.
# """

assistant_reply_format = """
Your job is to think, plan long-term goals and create list of tasks that will lead to you achieving the goal. Your plan should consist of small bulleted list that detail use of the tools you are given and how you plan to use the tool.
You should only respond in the XML format as described below.
Response Format:
<Start/>
<Thoughts>...</Thoughts>
<Reasoning>...</Reasoning>
<Plan>
<li> create lists using this tag</li>
</Plan>
<Criticism> constructive self criticism </Criticism>
<End/>
"""

class ToolboxAI:

  @staticmethod
  def call(
    model: Optional[OModel] = OModel.CHATGPT,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None
  ):
    tool_prompt = f'''
    You are StarfishGPT, a self-dependent autonomous Artificial Intelligence designed to accomplish complex goals assigned to you with no reliance on any human. You have access to the following tools that can help you ease your workflow:
    {demo_tools}

    Resources:
    1. Internet access for searching and gathering latest information
    2. Long-term memory
    3. Child agents to delegation of simple tasks

    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every tool you use has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

    {assistant_reply_format}
    '''
    sync_response = OAI.call(
      system_prompt=tool_prompt,
      prompt="""Goals:
      1. Search for latest news in India
      2. Group the articles by events
      3. Create a summary of each event citing sources
      4. Save the summary to a file
      5. Wait 30 mins
      6. Repeat""",
      stop_sequence=['<End/>'],
      stream=True
    )
    data = ""
    for response in sync_response:
      message = response["choices"][0]["delta"]
      if "content" in message:
        data += message["content"]
        print(message["content"], end="")
    return ""