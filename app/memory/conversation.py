import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Global memory saver checkpointer instance for thread persistence
memory_checkpointer = MemorySaver()


def get_conversation_history(graph_app: Any, thread_id: str) -> List[Dict[str, str]]:
    """
    Retrieve and format the existing conversation history for a given thread_id.
    
    Args:

        graph_app: Compiled LangGraph application instance.
        thread_id: Unique session identifier for the conversation.
        
    Returns:
        List of formatted message dictionaries: [{"role": "user/assistant", "content": "..."}]
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get state history from LangGraph checkpointer
        state = graph_app.get_state(config)
        
        if not state or "messages" not in state.values:
            return []
        
        formatted_history = []
        for message in state.values["messages"]:
            if isinstance(message, HumanMessage):
                formatted_history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_history.append({"role": "assistant", "content": message.content})
                
        return formatted_history

    except Exception as e:
        logger.error("Error retrieving conversation history for thread %s: %s", thread_id, e)
        return []


def clear_conversation_history(thread_id: str) -> bool:
    """
    Reset or wipe state checkpointer memory for a specific thread_id.
    
    Args:
        thread_id: Unique session identifier to clear.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # Overwrite state with empty messages list
        memory_checkpointer.put(config, checkpoint={"channel_values": {"messages": []}}, metadata={})
        logger.info("Successfully cleared conversation memory for thread: %s", thread_id)
        return True
    except Exception as e:
        logger.error("Failed to clear conversation history for thread %s: %s", thread_id, e)
        return False