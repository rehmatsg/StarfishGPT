import json
from duckduckgo_search import ddg


def search(query: str, num_results: int = 8) -> str:
  """Return the results of a Google search
  Args:
      query (str): The search query.
      num_results (int): The number of results to return.
  Returns:
      str: The results of the search.
  """
  search_results = []
  if not query:
    return json.dumps(search_results)

  results = ddg(query, max_results=num_results)
  if not results:
    return json.dumps(search_results)

  for j in results:
    search_results.append(j)

  return json.dumps(search_results, ensure_ascii=False, indent=4)

