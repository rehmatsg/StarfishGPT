import openai
from enum import Enum
from typing import Optional

class OAI:

  @staticmethod
  def call():
    return

  @staticmethod
  def __call_gpt3():
    return

  @staticmethod
  def __call_chatgpt(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list] = None
  ):
    assert (messages is None and prompt is not None and system_prompt is not None) or (messages is not None)
    if messages is None:
      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
      ]
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    return response

class OModel(Enum):
  GPT3 = "gpt3"
  CHATGPT = "chatgpt"