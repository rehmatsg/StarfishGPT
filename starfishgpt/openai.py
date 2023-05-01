import openai
from enum import Enum
from typing import Optional

class OModel(Enum):
  GPT3 = "gpt3"
  CHATGPT = "chatgpt"

class OAI:

  @staticmethod
  def call(
    model: Optional[OModel] = OModel.CHATGPT,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None,
    stop_sequence: Optional[list] = None,
    stream: Optional[bool] = False
  ):
    assert (
      (model == OModel.CHATGPT and ((messages is None and prompt is not None and system_prompt is not None) or messages is not None))
      or
      (model == OModel.GPT3 and (prompt is not None and system_prompt is not None))
    )
    if model == OModel.CHATGPT:
      return OAI.__call_chatgpt(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        stop_sequence=stop_sequence,
        stream=stream
      )
    else:
      return OAI.__call_gpt3(
        system_prompt=system_prompt,
        prompt=prompt,
        stop_sequence=stop_sequence,
        stream=stream
      )

  @staticmethod
  def __call_gpt3(
    prompt: str,
    system_prompt: Optional[str] = None,
    stop_sequence: Optional[list] = [],
    stream: Optional[bool] = False
  ):
    if system_prompt is not None:
      prompt = system_prompt + "\n" + prompt
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=2048,
      stop=stop_sequence,
      stream=stream
    )
    return response

  @staticmethod
  def __call_chatgpt(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None,
    stop_sequence: Optional[list] = [],
    stream: Optional[bool] = False
  ):
    assert (messages is None and prompt is not None and system_prompt is not None) or (messages is not None)
    if messages is None:
      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
      ]
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      stop=stop_sequence,
      stream=stream
    )
    return response