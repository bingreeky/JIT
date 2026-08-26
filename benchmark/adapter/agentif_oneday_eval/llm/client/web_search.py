"""
Web Search Client for fetching facts about questions and answers
Used as a fallback when web_search plugin causes 413 errors
"""
from typing import Dict, Any
from ...config import get_settings
from ...logging_config import get_logger
from .factory import LLMClientFactory
from ..schemas.types import LLMRequest, LLMMessage

logger = get_logger(__name__)


class WebSearchClient:
    """Web Search Client for fetching facts via LLM with web search enabled"""
    
    def __init__(self):
        """Initialize Web Search Client"""
        self.settings = get_settings()
    
    async def search_facts(
        self,
        question_text: str,
        answer_text: str,
        question_attachments: list = None,
        answer_attachments: list = None
    ) -> str:
        """
        Search for facts related to the question and answer
        
        Args:
            question_text: Question content
            answer_text: Answer content
            question_attachments: Question attachments
            answer_attachments: Answer attachments
            
        Returns:
            str: Fact summary from web search
        """
        # Build search prompt
        prompt = self._build_search_prompt(
            question_text,
            answer_text,
            question_attachments,
            answer_attachments
        )
        
        # Create LLM request with web search enabled
        llm_request = LLMRequest(
            model_name=self.settings.web_search_fallback_model,
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=self.settings.llm_max_tokens,
            temperature=0.1,
            use_grounding=True  # Enable web search
        )
        
        try:
            logger.info(f"Fetching web search facts using model: {self.settings.web_search_fallback_model}")
            logger.info(f"[WEB_SEARCH_CLIENT] Starting web search for question and answer facts")
            
            # Call LLM via factory
            client = LLMClientFactory.create_client(self.settings.web_search_fallback_model)
            response = await client.call(llm_request)
            
            # Trim result to max length
            result = response.content
            if len(result) > self.settings.web_search_fallback_max_length:
                logger.warning(
                    f"[WEB_SEARCH_CLIENT] Result too long ({len(result)} chars), "
                    f"trimming to {self.settings.web_search_fallback_max_length}"
                )
                result = result[:self.settings.web_search_fallback_max_length] + "\n\n[Result trimmed due to length]"
            
            logger.info(f"[WEB_SEARCH_CLIENT] Web search facts fetched successfully ({len(result)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"[WEB_SEARCH_CLIENT] Failed to fetch web search facts: {e}")
            return f"[Web search failed: {str(e)}]"
    
    def _build_search_prompt(
        self,
        question_text: str,
        answer_text: str,
        question_attachments: list = None,
        answer_attachments: list = None
    ) -> str:
        """Build prompt for web search"""
        
        prompt_parts = [
            "# Web Search Task",
            "",
            "## Objective",
            "Search the web and extract key factual information related to the following question and answer. "
            "Focus on verifiable facts, current information, and relevant context that can help verify the correctness of the answer.",
            "",
            "## Question"
        ]
        
        prompt_parts.append(question_text)
        
        # Add question attachments if available
        if question_attachments:
            prompt_parts.append("\n### Question Attachments")
            for i, att in enumerate(question_attachments, 1):
                prompt_parts.append(f"**Attachment {i}**: {att.get('filename', '')}")
                # Only include first 500 chars of attachment content to save space
                content = att.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                prompt_parts.append(content)
        
        prompt_parts.extend([
            "",
            "## Answer to Verify",
            answer_text
        ])
        
        # Add answer attachments if available
        if answer_attachments:
            prompt_parts.append("\n### Answer Attachments")
            for i, att in enumerate(answer_attachments, 1):
                prompt_parts.append(f"**Attachment {i}**: {att.get('filename', '')}")
                # Only include first 500 chars
                content = att.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                prompt_parts.append(content)
        
        prompt_parts.extend([
            "",
            "## Instructions",
            "1. Identify key claims, facts, dates, names, and other verifiable information in both the question and answer",
            "2. Use web search to verify these facts and gather relevant context",
            "3. Summarize the verified facts in a concise format",
            "4. Include:",
            "   - Verified factual information",
            "   - Current/updated information if relevant",
            "   - Any corrections to incorrect information",
            "   - Relevant context that helps assess answer quality",
            "",
            "## Output Format",
            "Provide a concise fact summary in the following format:",
            "",
            "**Verified Facts:**",
            "- [Fact 1]",
            "- [Fact 2]",
            "- ...",
            "",
            "**Additional Context:**",
            "- [Context 1]",
            "- [Context 2]",
            "- ...",
            "",
            "Keep the output concise and focused on information useful for scoring the answer."
        ])
        
        return "\n".join(prompt_parts)


