"""Helpers for normalizing citation payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def build_citation(*, url: str = "", source: str = "", page: Optional[int] = None) -> Dict[str, Any]:
	"""Return a normalized citation record."""
	return {
		"url": url,
		"source": source,
		"page": page,
	}


def dedupe_citations(citations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Deduplicate citation dictionaries while preserving order."""
	seen = set()
	normalized: List[Dict[str, Any]] = []

	for citation in citations:
		key = (
			citation.get("url", ""),
			citation.get("source", ""),
			citation.get("page"),
		)
		if key in seen:
			continue
		seen.add(key)
		normalized.append(
			{
				"url": citation.get("url", ""),
				"source": citation.get("source", ""),
				"page": citation.get("page"),
			}
		)

	return normalized
