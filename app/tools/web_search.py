from typing import List, Dict, Any
from app.config import settings
from app.utils.logger import logger

class WebSearchTool:
    """Wrapper around Tavily API for real-time web search with clean metadata return."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.api_key = settings.TAVILY_API_KEY

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Executes search query and returns structured results."""
        if not self.api_key:
            logger.warning("TAVILY_API_KEY is not configured. Web search will return empty results.")
            return []

        logger.info(f"Executing Tavily web search for query: '{query}'")
        try:
            # First attempt modern langchain-tavily integration
            try:
                from langchain_tavily import TavilySearch
                tool = TavilySearch(max_results=self.max_results)
                raw_results = tool.invoke({"query": query})
                
                # Format raw tool results into standard dictionary structure
                formatted_results = []
                if isinstance(raw_results, list):
                    for item in raw_results:
                        if isinstance(item, dict):
                            formatted_results.append({
                                "title": item.get("title", "No Title"),
                                "url": item.get("url", ""),
                                "content": item.get("content", item.get("snippet", "")),
                                "score": item.get("score", 0.0)
                            })
                return formatted_results
            except ImportError:
                # Fallback to direct tavily-python client
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.api_key)
                response = client.search(query=query, max_results=self.max_results)
                
                results = []
                for res in response.get("results", []):
                    results.append({
                        "title": res.get("title", "No Title"),
                        "url": res.get("url", ""),
                        "content": res.get("content", ""),
                        "score": res.get("score", 0.0)
                    })
                return results

        except Exception as e:
            logger.error(f"Error during web search execution: {str(e)}")
            return []