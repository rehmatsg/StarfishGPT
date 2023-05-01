import re
import json

class ResponseParser:

  @staticmethod
  def fromString(data: str):
    data = ResponseParser._get_fixed_string(data)
    return ResponseParser._extract_info(data)

  @staticmethod
  def _get_tags(text: str):
    open_tags = []
    close_tags = []
    pattern = r"<([a-zA-Z]+)[^>]*>"
    for match in re.findall(pattern, text):
      open_tags.append(match)
    pattern = r"</([a-zA-Z]+)[^>]*>"
    for match in re.findall(pattern, text):
      close_tags.append(match)
    return open_tags, close_tags
  
  @staticmethod
  def _get_unclosed_tags(open_tags, close_tags):
    unclosed_tags = []
    for tag in open_tags:
      if tag not in close_tags:
        unclosed_tags.append(tag)
    return unclosed_tags
  
  @staticmethod
  def _get_fixed_string(data: str):
    open_tags, close_tags = ResponseParser._get_tags(data)
    unclosed = ResponseParser._get_unclosed_tags(open_tags, close_tags)
    fixed_string = data
    for tag in reversed(unclosed):
      fixed_string += f"</{tag}>"
    return fixed_string
  
  @staticmethod
  def _extract_info(text: str):
    result = {}
    for tag in ["Thoughts", "Reasoning", "Plan", "Criticism", "Summary", "Tool"]:
      pattern = fr"<{tag}(?P<attrs>[^>]*)>(?P<values>[\s\S]*?)<\/{tag}>"
      match = re.search(pattern, text)
      if match:
        values = match.group("values").strip()
        attrs = match.group("attrs").strip()
        if attrs:
          attr_pattern = r'(?P<key>\w+)="(?P<value>[^"]*)"'
          parameters = {m.group('key'): m.group('value') for m in re.finditer(attr_pattern, attrs)}
        else:
          parameters = {}
        result[tag.lower()] = {"value": values, "parameters": parameters}
    return result