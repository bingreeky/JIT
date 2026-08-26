"""
Search and web crawling tools.
Adapted from Flash-Searcher FlashOAgents/search_tools.py
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import Tool

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


def read_page(url: str) -> str:
    """Read and return the content of a webpage using Jina reader."""
    jina_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown',
        "X-Remove-Selector": "header, .class, #id",
        "X-Retain-Images": "none",
        'X-Timeout': '10',
        'X-Token-Budget': '200000',
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error reading page: {str(e)}"


def web_search_google_serper(
    query: str,
    filter_year: Optional[int] = None,
    serp_num: int = 3,
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    """Perform web search using Google Serper API."""
    if not query.strip():
        return [], "Query is empty. Please provide a valid search query."

    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "location": "United States",
        "num": serp_num,
    })
    headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json',
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            results = response.json()

            if "organic" not in results or not results["organic"]:
                year_filter_msg = f" with year filter={filter_year}" if filter_year else ""
                return [], f"No results found for '{query}'{year_filter_msg}. Try a more general query."

            search_results = []
            for idx, page in enumerate(results["organic"], 1):
                search_results.append({
                    "idx": idx,
                    "title": page.get("title", "No title"),
                    "date": f"\nDate published: {page['date']}" if "date" in page else "",
                    "snippet": f"\n{page.get('snippet', 'No snippet')}",
                    "source": f"\nSource: {page.get('source', 'Unknown source')}",
                    "link": page.get('link', '#'),
                })

            return search_results, ""

        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                return [], f"Search failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)

    return [], "Unexpected error in web search"


class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Retrieve relevant knowledge from Wikipedia and return the search results."
    inputs = {
        "query": {
            "type": "string",
            "description": "Provide a query string for the information you want to retrieve from Wikipedia."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, query: str) -> str:
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
            "inprop": "url",
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'error' in data:
                error_info = data['error']
                return f"Wikipedia API error: {error_info.get('code', 'unknown')} - {error_info.get('info', 'unknown')}"

            pages = data.get("query", {}).get("pages", {})
            results = []

            for page_id, page_info in pages.items():
                if int(page_id) < 0:
                    continue
                title = page_info.get("title", "Unknown Title")
                extract = page_info.get("extract", "No extract available")
                page_url = page_info.get("fullurl", "No URL available")
                results.append(
                    f"[{title}]({page_url})\n"
                    f"Summary: {extract[:500]}{'...' if len(extract) > 500 else ''}"
                )

            return "\n\n".join(results) if results else f"No relevant information found for: {query}"

        except requests.Timeout:
            return "Request to Wikipedia API timed out. Please try again later."
        except requests.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


class WebSearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query and return the search results."
    inputs = {
        "query": {
            "type": "string",
            "description": "The web search query to perform."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, query: str) -> str:
        search_results, error_msg = web_search_google_serper(query, serp_num=5)

        if error_msg:
            return error_msg

        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"{result['idx']}. [{result['title']}]({result['link']})"
                f"{result['date']}{result['source']}\n"
                f"   {result['snippet'].strip()}"
            )

        return "\n\n".join(formatted_results) if formatted_results else "No search results found"


class CrawlPageTool(Tool):
    name = "crawl_page"
    description = "Access webpage using the provided URL and extract relevant content. Please make full use of this tool to verify the accuracy of the searched content."
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit."
        },
        "query": {
            "type": "string",
            "description": "The specific information to extract from the webpage."
        }
    }
    output_type = "string"

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    @staticmethod
    def truncate_text(text: str, max_length: int = 60000) -> str:
        return text if len(text) <= max_length else text[:max_length] + "...(truncated)"

    @staticmethod
    def _parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
        text = str(raw_text or "").strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _build_extract_messages(self, query: str, url: str, content: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a webpage evidence extractor.\n"
            "Return STRICT JSON only with keys:\n"
            "- status: one of found|not_found|error\n"
            "- answer: concise answer text\n"
            "- relevant_points: array of concise bullet-like strings\n"
            "- evidence_quotes: array of short verbatim quotes from page text\n\n"
            "Rules:\n"
            "- If information is missing, set status=not_found and answer exactly 'No relevant information'.\n"
            "- Do not invent data not present in content.\n"
            "- Prefer exact values/dates/numbers when present.\n"
            "- Keep answer and points concise."
        )
        user_prompt = (
            f"Query:\n{query}\n\n"
            f"URL:\n{url}\n\n"
            f"Web page text:\n{content}\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_with_model(self, query: str, url: str, content: str, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        if not callable(self.model):
            return None
        messages = self._build_extract_messages(query=query, url=url, content=content)

        for attempt in range(max_retries):
            try:
                response = self.model(messages)
                raw = response.content if hasattr(response, "content") else str(response)
                obj = self._parse_json_object(raw)
                if isinstance(obj, dict):
                    return obj

                # One repair attempt when model returns non-JSON text.
                repair_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. "
                            "Return only one valid JSON object with keys "
                            "status, answer, relevant_points, evidence_quotes."
                        ),
                    }
                ]
                repair_resp = self.model(repair_messages)
                repair_raw = (
                    repair_resp.content if hasattr(repair_resp, "content") else str(repair_resp)
                )
                repaired = self._parse_json_object(repair_raw)
                if isinstance(repaired, dict):
                    return repaired
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        return None

    def _format_structured_result(self, obj: Dict[str, Any]) -> str:
        status = str(obj.get("status", "")).strip().lower()
        answer = self._normalize_spaces(obj.get("answer", ""))

        if status == "not_found" or answer.lower() == "no relevant information":
            return "No relevant information"
        if status == "error":
            return "No relevant information"

        points = obj.get("relevant_points", [])
        if not isinstance(points, list):
            points = []
        cleaned_points = []
        for x in points[:8]:
            line = self._normalize_spaces(x)
            if line:
                cleaned_points.append(f"- {line}")

        if answer:
            if cleaned_points:
                return answer + "\n\n" + "\n".join(cleaned_points)
            return answer

        if cleaned_points:
            return "\n".join(cleaned_points)
        return "No relevant information"

    def _heuristic_extract(self, query: str, content: str, max_lines: int = 8) -> str:
        # Lightweight fallback when the model cannot produce valid JSON.
        q_tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
        if not q_tokens:
            q_tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower())]

        lines = [self._normalize_spaces(ln) for ln in content.splitlines() if self._normalize_spaces(ln)]
        scored: List[Tuple[int, str]] = []
        for ln in lines:
            low = ln.lower()
            score = sum(1 for tok in q_tokens if tok in low)
            if score > 0:
                scored.append((score, ln))

        if not scored:
            return "No relevant information"

        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        seen = set()
        picked = []
        for _, ln in scored:
            if ln in seen:
                continue
            seen.add(ln)
            picked.append(f"- {ln}")
            if len(picked) >= max_lines:
                break

        return "\n".join(picked) if picked else "No relevant information"

    def forward(self, url: str, query: str) -> str:
        if not url.startswith(('http://', 'https://')):
            return "Invalid URL format. Must start with http:// or https://"

        page_content = read_page(url)
        if page_content.startswith("Error"):
            return page_content

        truncated_content = self.truncate_text(page_content)
        extracted = self._extract_with_model(query=query, url=url, content=truncated_content)
        if isinstance(extracted, dict):
            return self._format_structured_result(extracted)

        return self._heuristic_extract(query=query, content=truncated_content)


__all__ = ["WikiSearchTool", "WebSearchTool", "CrawlPageTool"]
