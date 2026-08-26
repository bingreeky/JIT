"""
Async Scoring Executor
Supports concurrency control, failure retry, and progress tracking
"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from .config import get_settings
from .logging_config import get_logger
from .data_loader import Question, Answer, ScoreResult, DataLoader, AttachmentLoader
from .llm.client.factory import LLMClientFactory
from .llm.client.openrouter import PayloadTooLargeError, ContextLengthExceededError
from .llm.client.web_search import WebSearchClient
from .llm.schemas.types import LLMRequest, LLMMessage, ImageAttachment
from .llm.prompts.batch_score.prompt_template import build_batch_score_prompt
from .llm.utils.attachment_parser import AttachmentParser, ParsedAttachmentResult

logger = get_logger(__name__)


@dataclass
class ScoringTask:
    """Scoring Task"""
    question: Question
    answer: Answer
    model_name: str


@dataclass
class ScoringProgress:
    """Scoring Progress"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retry_count: int = 0
    current_task: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class AsyncScorer:
    """Async Scoring Executor"""
    
    def __init__(
        self,
        model_name: str,
        max_concurrent: int = 5,
        max_retries: int = 3,
        attachment_base_path: str = "Attachments"
    ):
        """
        Initialize scoring executor
        
        Args:
            model_name: LLM model name to use
            max_concurrent: Maximum concurrent requests
            max_retries: Maximum retry attempts
            attachment_base_path: Attachment base path
        """
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.attachment_loader = AttachmentLoader(attachment_base_path)
        self.attachment_parser = AttachmentParser()
        self.web_search_client = WebSearchClient()
        self.progress = ScoringProgress()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._write_lock: Optional[asyncio.Lock] = None  # File write lock
    
    async def score_all(
        self,
        questions: List[Question],
        answers: List[Answer],
        output_file: str,
        skip_existing: bool = False,
        progress_callback: Optional[Callable[[ScoringProgress], None]] = None
    ) -> List[ScoreResult]:
        """
        Score all question-answer pairs
        
        Args:
            questions: List of questions
            answers: List of answers
            output_file: Output file path
            skip_existing: Deprecated, kept for compatibility
            progress_callback: Progress callback function
            
        Returns:
            List of scoring results
        """
        # Build question dictionary
        question_dict = {q.question_id: q for q in questions}
        
        # Build scoring task list
        tasks = []
        for answer in answers:
            question = question_dict.get(answer.question_id)
            if not question:
                logger.warning(f"Question {answer.question_id} not found, skipping answer")
                continue
            
            tasks.append(ScoringTask(
                question=question,
                answer=answer,
                model_name=self.model_name
            ))
        
        if not tasks:
            logger.info("No scoring tasks to process")
            return []
        
        # Initialize progress
        self.progress = ScoringProgress(
            total_tasks=len(tasks),
            start_time=datetime.now()
        )
        
        logger.info(f"Starting scoring: {len(tasks)} tasks, max concurrent {self.max_concurrent}")
        
        # Create semaphore to control concurrency
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        # Create file write lock
        self._write_lock = asyncio.Lock()
        
        # Execute scoring tasks concurrently
        all_results = []
        
        async def process_task(task: ScoringTask) -> List[ScoreResult]:
            async with self._semaphore:
                results = await self._score_single_task(task)
                
                # Save results immediately (save after each task completes)
                # Use write lock to protect file operations, ensuring atomicity and ordering
                if results:
                    async with self._write_lock:
                        DataLoader.save_score_results(results, output_file, append=True)
                
                # Update progress
                self.progress.completed_tasks += 1
                self.progress.current_task = f"{task.question.question_id} - {task.answer.agent_name}"
                
                if progress_callback:
                    progress_callback(self.progress)
                
                return results
        
        # Use asyncio.gather to execute all tasks concurrently
        task_results = await asyncio.gather(
            *[process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Collect results
        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Task execution error: {result}")
                self.progress.failed_tasks += 1
            elif result:
                all_results.extend(result)
        
        # Complete
        self.progress.end_time = datetime.now()
        duration = (self.progress.end_time - self.progress.start_time).total_seconds()
        
        logger.info(
            f"Scoring completed: success {self.progress.completed_tasks}/{self.progress.total_tasks}, "
            f"failed {self.progress.failed_tasks}, "
            f"retries {self.progress.retry_count}, "
            f"duration {duration:.1f} seconds"
        )
        
        return all_results
    
    async def _score_single_task(self, task: ScoringTask) -> List[ScoreResult]:
        """
        Execute single scoring task (with retry)
        
        Args:
            task: Scoring task
            
        Returns:
            List of scoring results
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                if retry_count > 0:
                    logger.warning(
                        f"Retrying scoring (attempt {retry_count}): "
                        f"{task.question.question_id} - {task.answer.agent_name}"
                    )
                    # Exponential backoff
                    await asyncio.sleep(2 ** retry_count)
                    self.progress.retry_count += 1
                
                # Execute scoring
                results = await self._do_score(task)
                
                if retry_count > 0:
                    logger.info(
                        f"Retry successful: {task.question.question_id} - {task.answer.agent_name}"
                    )
                else:
                    logger.info(
                        f"Scoring completed: {task.question.question_id} - {task.answer.agent_name}"
                    )
                
                return results
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Check if should retry
                if not self._should_retry(e) or retry_count >= self.max_retries:
                    logger.error(
                        f"Scoring failed (after {retry_count-1} retries): "
                        f"{task.question.question_id} - {task.answer.agent_name} - {e}"
                    )
                    break
        
        # All retries failed, return empty result
        return []
    
    async def _do_score(self, task: ScoringTask) -> List[ScoreResult]:
        """
        Execute actual scoring operation
        
        Args:
            task: Scoring task
            
        Returns:
            List of scoring results
        """
        settings = get_settings()
        
        # Get attachment content
        question_attachments = await self._get_question_attachments(task.question)
        # Answer attachments now returns both text and image attachments
        answer_text_attachments, answer_image_attachments = await self._get_answer_attachments(task.answer)
        reference_answer_attachments = await self._get_reference_answer_attachments(task.question)
        
        # Build scoring criteria
        score_criteria = [
            {
                "criterion_id": criterion.criterion_id or str(i),
                "content": criterion.content,
                "score": criterion.score
            }
            for i, criterion in enumerate(task.question.score_criteria, 1)
        ]
        
        # Check if grounding is enabled
        use_grounding = False
        if task.model_name.startswith("gemini-") and settings.enable_google_search_grounding:
            use_grounding = True
        
        # Get answer text
        answer_text = task.answer.content.get('text', '') if isinstance(task.answer.content, dict) else str(task.answer.content)
        
        # Try normal scoring first
        try:
            return await self._do_score_with_llm(
                task=task,
                question_attachments=question_attachments,
                answer_text=answer_text,
                answer_text_attachments=answer_text_attachments,
                answer_image_attachments=answer_image_attachments,
                reference_answer_attachments=reference_answer_attachments,
                score_criteria=score_criteria,
                use_grounding=use_grounding
            )
        except (PayloadTooLargeError, ContextLengthExceededError) as e:
            # If 413 error or context length exceeded, and fallback is enabled, try fallback approach
            if settings.web_search_fallback_enabled:
                error_type = "Payload too large (413)" if isinstance(e, PayloadTooLargeError) else "Context length exceeded (400)"
                logger.warning(
                    f"[WEB_SEARCH_FALLBACK] {error_type} detected for {task.question.question_id} - {task.answer.agent_name}"
                )
                logger.info(
                    f"[WEB_SEARCH_FALLBACK] Activating web search fallback strategy for {task.question.question_id} - {task.answer.agent_name}"
                )
                return await self._do_score_with_fallback(
                    task=task,
                    question_attachments=question_attachments,
                    answer_text=answer_text,
                    answer_text_attachments=answer_text_attachments,
                    answer_image_attachments=answer_image_attachments,
                    reference_answer_attachments=reference_answer_attachments,
                    score_criteria=score_criteria
                )
            else:
                # Fallback disabled, re-raise the error
                logger.warning(f"Web search fallback is disabled, re-raising error: {e}")
                raise
    
    async def _do_score_with_llm(
        self,
        task: ScoringTask,
        question_attachments: List[Dict[str, str]],
        answer_text: str,
        answer_text_attachments: List[Dict[str, str]],
        answer_image_attachments: List[ImageAttachment],
        reference_answer_attachments: List[Dict[str, str]],
        score_criteria: List[Dict[str, Any]],
        use_grounding: bool,
        web_search_facts: str = ""
    ) -> List[ScoreResult]:
        """
        Execute scoring with LLM call
        
        Args:
            task: Scoring task
            question_attachments: Question attachments
            answer_text: Answer text
            answer_text_attachments: Answer text attachments
            answer_image_attachments: Answer image attachments
            reference_answer_attachments: Reference answer attachments
            score_criteria: Scoring criteria
            use_grounding: Whether to use web search grounding
            web_search_facts: Pre-fetched web search facts (optional)
            
        Returns:
            List of scoring results
        """
        settings = get_settings()
        
        # Build prompt
        prompt = build_batch_score_prompt(
            question_text=task.question.description or task.question.title,
            question_attachments=question_attachments,
            answer_text=answer_text,
            answer_attachments=answer_text_attachments,
            score_criteria=score_criteria,
            use_grounding=use_grounding,
            reference_answer_text=task.question.reference_answer_description,
            reference_answer_attachments=reference_answer_attachments,
            web_search_facts=web_search_facts
        )
        
        # Log if we have image attachments
        if answer_image_attachments:
            logger.info(f"Scoring with {len(answer_image_attachments)} image attachment(s) for {task.question.question_id}")
        
        # Call LLM with image attachments
        llm_request = LLMRequest(
            model_name=task.model_name,
            messages=[LLMMessage(role="user", content=prompt)],
            image_attachments=answer_image_attachments if answer_image_attachments else None,
            max_tokens=settings.llm_max_tokens,
            temperature=0.0,
            use_grounding=use_grounding
        )
        
        llm_response = await LLMClientFactory.call_with_attachments(llm_request)
        
        # Parse response
        llm_result = self._parse_llm_response(llm_response.content)
        
        # Build scoring results
        results = []
        for criterion in task.question.score_criteria:
            # Find corresponding LLM result
            llm_criterion_result = None
            criterion_id = criterion.criterion_id or str(task.question.score_criteria.index(criterion) + 1)
            
            for result in llm_result.get('criteria_results', []):
                if result.get('criterion_id') == criterion_id:
                    llm_criterion_result = result
                    break
            
            score_result = ScoreResult(
                question_id=task.question.question_id,
                agent_name=task.answer.agent_name,
                method=task.model_name,
                criterion_content=criterion.content,
                criterion_score=criterion.score,
                satisfied=llm_criterion_result.get('satisfied', False) if llm_criterion_result else False,
                reasoning=llm_criterion_result.get('reasoning', '') if llm_criterion_result else ''
            )
            results.append(score_result)
        
        return results
    
    async def _do_score_with_fallback(
        self,
        task: ScoringTask,
        question_attachments: List[Dict[str, str]],
        answer_text: str,
        answer_text_attachments: List[Dict[str, str]],
        answer_image_attachments: List[ImageAttachment],
        reference_answer_attachments: List[Dict[str, str]],
        score_criteria: List[Dict[str, Any]]
    ) -> List[ScoreResult]:
        """
        Execute scoring with web search fallback
        
        First fetches facts via web search, then calls LLM without web_search plugin
        
        Args:
            task: Scoring task
            question_attachments: Question attachments
            answer_text: Answer text
            answer_text_attachments: Answer text attachments
            answer_image_attachments: Answer image attachments
            reference_answer_attachments: Reference answer attachments
            score_criteria: Scoring criteria
            
        Returns:
            List of scoring results
        """
        logger.info(
            f"[WEB_SEARCH_FALLBACK] Step 1: Fetching web search facts for "
            f"{task.question.question_id} - {task.answer.agent_name}"
        )
        
        question_text = task.question.description or task.question.title
        web_search_facts = await self.web_search_client.search_facts(
            question_text=question_text,
            answer_text=answer_text,
            question_attachments=question_attachments,
            answer_attachments=answer_text_attachments
        )
        
        logger.info(
            f"[WEB_SEARCH_FALLBACK] Step 2: Web search facts obtained ({len(web_search_facts)} chars) for "
            f"{task.question.question_id} - {task.answer.agent_name}"
        )
        
        # Step 2: Score with LLM without web_search plugin, but with pre-fetched facts
        logger.info(
            f"[WEB_SEARCH_FALLBACK] Step 3: Scoring with pre-fetched facts (web_search=False) for "
            f"{task.question.question_id} - {task.answer.agent_name}"
        )
        
        results = await self._do_score_with_llm(
            task=task,
            question_attachments=question_attachments,
            answer_text=answer_text,
            answer_text_attachments=answer_text_attachments,
            answer_image_attachments=answer_image_attachments,
            reference_answer_attachments=reference_answer_attachments,
            score_criteria=score_criteria,
            use_grounding=False,  # Disable web_search plugin
            web_search_facts=web_search_facts
        )
        
        logger.info(
            f"[WEB_SEARCH_FALLBACK] Completed successfully for "
            f"{task.question.question_id} - {task.answer.agent_name}"
        )
        
        return results
    
    async def _get_question_attachments(self, question: Question) -> List[Dict[str, str]]:
        """Get question attachment content"""
        attachments = []
        
        for filename in question.attachment_filenames:
            path = self.attachment_loader.get_question_attachment_path(filename)
            if path:
                try:
                    content = await self.attachment_parser.parse_single_attachment(path)
                    attachments.append({
                        "filename": filename,
                        "content": content
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse question attachment {filename}: {e}")
                    attachments.append({
                        "filename": filename,
                        "content": f"[Attachment parsing failed: {e}]"
                    })
            else:
                logger.warning(f"Question attachment does not exist: {filename}")
        
        return attachments
    
    async def _get_answer_attachments(self, answer: Answer) -> tuple:
        """
        Get answer attachment content and screenshots
        
        Returns:
            Tuple of (text_attachments, image_attachments)
            - text_attachments: List[Dict[str, str]] with "filename" and "content"
            - image_attachments: List[ImageAttachment] for multimodal models
        """
        text_attachments = []
        image_attachments = []
        
        # Deduplicate attachment list to avoid repeated processing
        unique_filenames = list(dict.fromkeys(answer.attachment_filenames))
        
        for filename in unique_filenames:
            path = self.attachment_loader.get_answer_attachment_path(answer.agent_name, filename)
            if path:
                try:
                    # Use new method that returns both text and screenshots
                    result: ParsedAttachmentResult = await self.attachment_parser.parse_attachment_with_screenshots(path)
                    
                    # Add text content
                    if result.text_content:
                        text_attachments.append({
                            "filename": filename,
                            "content": result.text_content
                        })
                    
                    # Add any screenshots as image attachments
                    for screenshot in result.screenshots:
                        image_attachments.append(ImageAttachment(
                            filename=screenshot.get("filename", filename),
                            mime_type=screenshot.get("mime_type", "image/png"),
                            base64_data=screenshot.get("base64_data", "")
                        ))
                        logger.info(f"Added screenshot for {filename}: {screenshot.get('filename')}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse answer attachment {filename}: {e}")
                    text_attachments.append({
                        "filename": filename,
                        "content": f"[Attachment parsing failed: {e}]"
                    })
            else:
                logger.warning(f"Answer attachment does not exist: {answer.agent_name}/{filename}")
        
        return text_attachments, image_attachments
    
    async def _get_reference_answer_attachments(self, question: Question) -> List[Dict[str, str]]:
        """Get reference answer attachment content"""
        attachments = []
        
        # Deduplicate attachment list to avoid repeated processing
        unique_filenames = list(dict.fromkeys(question.reference_answer_attachment_filenames))
        
        for filename in unique_filenames:
            path = self.attachment_loader.get_reference_answer_attachment_path(filename)
            if path:
                try:
                    content = await self.attachment_parser.parse_single_attachment(path)
                    attachments.append({
                        "filename": filename,
                        "content": content
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse reference answer attachment {filename}: {e}")
                    attachments.append({
                        "filename": filename,
                        "content": f"[Attachment parsing failed: {e}]"
                    })
            else:
                logger.warning(f"Reference answer attachment does not exist: {filename}")
        
        return attachments
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            # Try to extract JSON from markdown code block
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            elif "```" in response_content:
                json_start = response_content.find("```") + 3
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content.strip()
            
            return json.loads(json_content)
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"criteria_results": []}
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if error should be retried"""
        error_msg = str(exception).lower()
        
        # Retryable error patterns
        retryable_patterns = [
            'connection', 'timeout', 'timed out', 'remote end closed',
            '429', '500', '502', '503'
        ]
        
        for pattern in retryable_patterns:
            if pattern in error_msg:
                return True
        
        # Non-retryable errors
        non_retryable_patterns = [
            '401', '403', '404', 'unauthorized', 'forbidden', 'not found'
        ]
        
        for pattern in non_retryable_patterns:
            if pattern in error_msg:
                return False
        
        # ValueError is usually a data issue, should not retry
        if isinstance(exception, ValueError):
            return False
        
        # Default to retry for other types of errors
        return True
