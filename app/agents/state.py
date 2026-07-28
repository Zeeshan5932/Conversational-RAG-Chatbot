from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Represents the shared state of the LangGraph execution flow."""

    # Chat history and messages
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Query details
    user_query: str
    route_decision: Optional[str]
    
    # Context collected during execution
    retrieved_docs: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    url_content: Optional[str]
    
    # Final output
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]