from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.llm.gemini import get_gemini_llm
from app.utils.logger import logger


class QueryRoute(BaseModel):
    """Structured classification output for routing user queries."""

    route: Literal["general", "web_search", "document_rag", "url_research", "hybrid"] = Field(
        description="The target handler for the query based on intent."
    )
    reasoning: str = Field(
        description="Short justification for why this route was selected."
    )


class QueryRouter:
    """Classifies user queries to determine the optimal execution path."""

    def __init__(self):
        self.llm = get_gemini_llm(temperature=0.0)
        self.structured_llm = self.llm.with_structured_output(QueryRoute)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for a research assistant agent.
Analyze the user query and classify it into exactly ONE of the following routing choices:

1. 'general': Conversational messages, greetings, simple logic/math, coding questions, or general queries not requiring outside knowledge or personal documents.
2. 'document_rag': Questions specifically referring to uploaded files, CVs, PDFs, internal documents, or personal files.
3. 'web_search': Questions asking for real-time information, recent events, breaking news, live data, or current trends.
4. 'url_research': Explicit requests to summarize, read, or analyze a specific URL string (e.g., http:// or https://).
5. 'hybrid': Complex queries requiring BOTH information from uploaded documents AND up-to-date web search comparison.

Analyze the query carefully before making your selection."""),
            ("human", "Query: {query}")
        ])

    def route_query(self, query: str) -> QueryRoute:
        """Invokes structured output chain to route user prompt."""
        logger.info(f"Routing user query: '{query}'")
        chain = self.prompt | self.structured_llm
        try:
            result: QueryRoute = chain.invoke({"query": query})
            logger.info(f"Query routed to: [{result.route}] | Reason: {result.reasoning}")
            return result
        except Exception as e:
            logger.error(f"Routing failed with error: {str(e)}. Defaulting to 'general'.")
            return QueryRoute(route="general", reasoning="Fallback due to router exception.")