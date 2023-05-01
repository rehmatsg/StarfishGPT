from starfishgpt.openai import OAI, OModel
from typing import Optional
import re
from typing import Tuple

demo_tools = [
  {
    "name": "search",
    "usage": "search(query: str) -> list",
    "description": "Issues a query to a search engine and returns a list of results. Each result is a Python Dictionary with the following fields: [title, url]",
    "module": "starfishgpt.tools"
  }
]

_main_prompt = """
You are StarExecuter, an AI agent working under the supervision of another AI agent. You are given a prompt and you need to generate Python code to accomplish the task provided. While generating the code, follow these rules:

1. Output only the main code block without any usage examples or additional text.
2. Always include a docstring at the beginning of the `main()` function that describes the function and each argument.

You are given access to certain tools (Python functions) that you can use in your code:

{tools}

To use a tool, you will first need to import it from the specified module:

`from [starfishgpt.tools] import execute_shell_command`

Then, you can continue to use the tool as a Python function:

`ouput = execute_shell_command("ls")`

Performance Evaluation:
1. Every tool has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You generate answers, but do not want to engage the user in any way, including explaining your work, giving further instructions, or asking for clarification.

There is a format you need to follow to create Python scripts. Each Python script will contain a `main()` function that can be called to run that script. You can add any number of arguments to the function. The main function must always return a final result. If the function needed is a void function, you can return the status as a string (e.g. "Task executed successfully"). Additionally, you will always include a docstring that describes the function, each argument and the return type.

You will also be including a filename and a pip command at the beginning of the reponse that tells the code interpreter to install external packages that will be used. If you do not require any external packages, you can skip this.

You will be working independently of the user. You cannot ask the user to help you at any step. You can add any number of arguments to the function for all the variables you need to accomplish the task.

Following is the format that you will be using to output:

```python
#filename.py
!pip install [comma separated package names]
def main(*args) -> str:
  '''
  Function description.
    
  Args:
    arg1 (type): Description of arg1.
    arg2 (type): Description of arg2.
    ...
    
  Returns:
    str: Final result or status message.
  '''
  ## Here comes your code
  return final_result
```
"""

_fix_error_prompt = """
The code you generated encountered a few errors. You need to regenerate the code and output in the same format. Make sure that the updated code fixes the errors. Given below is the error message:
{error_message}
"""

_safety_check_prompt = """
You are StarfishGPT, an expert programmer. One of a new programmers has generated some Python code that need to be run on a live server. Your task is to check if the code can generate any harm or might prove to be destructive.

You will be following a certain format to output.
<SafetyCheck safe="a boolean value here" />

If the code is safe, you will set the "safe" attribute to True. If the code is safe but requires caution while running, add an additional attribute "caution" and set it's value to a string with a reason.
Finally, if the code is unsafe, set the "safe" attribute to False and add an additional attribute "reason" and set it's value to a string with a single sentence reason why it's unsafe.

You will be working independently of the user. You cannot ask the user to help you at any step. You can add any number of arguments to the function for all the variables you need to accomplish the task.

You will strictly follow the response format and only output the XML format defined. Make sure your response is parsable by XML parsers.
"""

class TaskMaster:

  def __init__(
    self,
    prompt: str,
    model: OModel,
    raw_response: str,
    filename: str,
    packages: list[str],
    code: str,
    is_safe: bool,
    caution: str | None,
    unsafe_reason: str | None,
  ):
    self.filename = filename
    self.packages = packages
    self.code = code
    self.is_safe = is_safe
    self.caution = caution
    self.unsafe_reason = unsafe_reason
    self.message_history = [
      {
        "role": "system",
        "content": _main_prompt.format(tools=demo_tools)
      },
      {
        "role": "user",
        "content": prompt
      },
      {
        "role": "assistant",
        "content": raw_response
      }
    ]

  @classmethod
  def create(
    cls,
    prompt: Optional[str] = None,
    model: Optional[OModel] = OModel.CHATGPT,
  ):
    response = TaskMaster._generate(prompt, model)
    filename, packages, code = TaskMaster._extract(response)
    is_safe, caution, unsafe_reason = TaskMaster._safety_check(code)
    return cls(prompt, model, response, filename, packages, code, is_safe, caution, unsafe_reason)

  @staticmethod
  def _generate(
    prompt: Optional[str] = None,
    model: Optional[OModel] = OModel.CHATGPT,
  ):
    response = OAI.call(
      system_prompt=_main_prompt.format(tools=demo_tools),
      prompt=prompt,
      stream=False
    )
    return response["choices"][0]["message"]["content"]

  @staticmethod
  def _extract(input_string: str):
    """
    This function extracts Python code, filename and pip packages from the GPT's response.
    """
    # Remove the triple backticks (```python and ```)
    input_string = re.sub(r'^```python|```$', '', input_string, flags=re.MULTILINE).strip()

    # Extract filename
    filename_match = re.search(r'^# ?(.+\.py)$', input_string, flags=re.MULTILINE)
    if filename_match:
      filename = filename_match.group(1).strip()
    else:
      raise ValueError("Filename not included or unable to parse the filename")

    # Extract pip packages
    pip_packages = []
    for match in re.finditer(r'^!pip install (.+)$', input_string, flags=re.MULTILINE):
      pip_packages.extend([pkg.strip() for pkg in match.group(1).split(',')])

    # Extract python code
    python_code = re.sub(r'^(# ?.+\.py)|(!pip.+)', '', input_string, flags=re.MULTILINE).strip()

    # Ensure filename, pip, and python functions are in order
    if re.search(r'^(# ?.+\.py)(.+)?(!pip.+)', input_string) or re.search(r'^(# ?.+\.py)(.+)?(^def .+)', input_string, flags=re.MULTILINE):
      raise ValueError("Invalid order of filename, pip, and python functions")

    return filename, pip_packages, python_code

  @staticmethod
  def _safety_check(code: str) -> Tuple[bool, str | None, str | None]:
    response = OAI.call(
      system_prompt=_safety_check_prompt,
      prompt=code,
      stream=False
    )

    message = response["choices"][0]["message"]["content"]
    
    safe_pattern = re.compile(r'safe="(.*?)"')
    caution_pattern = re.compile(r'caution="(.*?)"')
    reason_pattern = re.compile(r'reason="(.*?)"')
    
    safe = safe_pattern.search(message)
    caution = caution_pattern.search(message)
    reason = reason_pattern.search(message)

    is_safe = safe.group(1).lower() == "true" if safe else False
    caution = caution.group(1) if caution else None
    reason = reason.group(1) if reason else None
    return is_safe, caution, reason

  def run(self, *args, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
      try:
        funcs = {}
        exec(self.code, funcs)
        main = funcs["main"]
        output = main(*args)  # Pass the arguments using the unpacking operator
        if output is None:
          return f"{self.filename} run successfully"
        else:
          return output
      except Exception as e:
        print(f"An exception occurred (attempt {retry_count + 1}):", str(e))
        self._recreate_after_error(str(e))
        retry_count += 1
    return f"Failed to execute {self.filename} after {max_retries} attempts"

  def _recreate_after_error(
    self,
    error: str
  ):
    self.message_history.append(
      {
        "role": "user",
        "content": _fix_error_prompt.format(error_message=error)
      }
    )
    response = OAI.call(
      model=self.model,
      messages=self.message_history,
      stream=False
    )
    message = response["choices"][0]["message"]["content"]
    self.message_history.append(
      {
        "role": "assistant",
        "content": message
      }
    )
    filename, packages, code = TaskMaster._extract(message)
    print("Fixed Code:\n" + code)
    self.code = code
    # is_safe, caution, unsafe_reason = TaskMaster._safety_check(code)

  def __str__(self):
    return f"Filename: {self.filename}\nPackages: {' '.join(self.packages)}\nCode:\n{self.code}\nIs Safe: {self.is_safe}\nCaution: {self.caution}\nUnsafe Reason: {self.unsafe_reason}"