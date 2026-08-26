"""
JSONL Data Loader
For loading questions, answers and outputting scoring results
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ScoreCriterion:
    """Scoring Criterion"""
    content: str
    score: int
    criterion_id: Optional[str] = None


@dataclass 
class Question:
    """Question Data"""
    question_id: str
    title: str
    description: str
    attachment_filenames: List[str] = field(default_factory=list)
    score_criteria: List[ScoreCriterion] = field(default_factory=list)
    reference_answer_description: str = ""
    reference_answer_attachment_filenames: List[str] = field(default_factory=list)


@dataclass
class Answer:
    """Answer Data"""
    question_id: str
    agent_name: str
    content: Dict[str, Any]
    attachment_filenames: List[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    """Scoring Result"""
    question_id: str
    agent_name: str
    method: str
    criterion_content: str
    criterion_score: int
    satisfied: bool
    reasoning: str


class DataLoader:
    """JSONL Data Loader"""
    
    def __init__(self, attachment_base_path: str = "Attachments"):
        """
        Initialize data loader
        
        Args:
            attachment_base_path: Base path for attachments
        """
        self.attachment_base_path = Path(attachment_base_path)
    
    def load_questions(self, file_path: str) -> List[Question]:
        """
        Load questions from JSONL file
        
        Args:
            file_path: JSONL file path
            
        Returns:
            List of questions
        """
        questions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Parse scoring criteria
                    score_criteria = []
                    for criterion_data in data.get('score_criteria', []):
                        criterion = ScoreCriterion(
                            content=criterion_data.get('content', ''),
                            score=criterion_data.get('score', 0),
                            criterion_id=criterion_data.get('criterion_id')
                        )
                        score_criteria.append(criterion)
                    
                    question = Question(
                        question_id=data.get('question_id', ''),
                        title=data.get('title', ''),
                        description=data.get('description', ''),
                        attachment_filenames=data.get('attachment_filenames', []),
                        score_criteria=score_criteria,
                        reference_answer_description=data.get('reference_answer_description', ''),
                        reference_answer_attachment_filenames=data.get('reference_answer_attachment_filenames', [])
                    )
                    questions.append(question)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse question at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(questions)} questions from {file_path}")
        return questions
    
    def load_answers(self, file_path: str) -> List[Answer]:
        """
        Load answers from JSONL file
        
        Args:
            file_path: JSONL file path
            
        Returns:
            List of answers
        """
        answers = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    answer = Answer(
                        question_id=data.get('question_id', ''),
                        agent_name=data.get('agent_name', ''),
                        content=data.get('content', {}),
                        attachment_filenames=data.get('attachment_filenames', [])
                    )
                    answers.append(answer)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse answer at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(answers)} answers from {file_path}")
        return answers
    
    def get_question_attachment_paths(self, question: Question) -> List[str]:
        """
        Get full paths for question attachments
        
        Args:
            question: Question object
            
        Returns:
            List of attachment paths
        """
        paths = []
        for filename in question.attachment_filenames:
            # Question attachments are stored in "Attachments/Questions" directory
            path = self.attachment_base_path / "Questions" / filename
            if path.exists():
                paths.append(str(path))
            else:
                logger.warning(f"Question attachment does not exist: {path}")
        return paths
    
    def get_answer_attachment_paths(self, answer: Answer) -> List[str]:
        """
        Get full paths for answer attachments
        
        Args:
            answer: Answer object
            
        Returns:
            List of attachment paths
        """
        paths = []
        for filename in answer.attachment_filenames:
            # Answer attachments are stored in "Attachments/{agent_name}" directory
            path = self.attachment_base_path / answer.agent_name / filename
            if path.exists():
                paths.append(str(path))
            else:
                logger.warning(f"Answer attachment does not exist: {path}")
        return paths
    
    @staticmethod
    def save_score_results(results: List[ScoreResult], file_path: str, append: bool = True):
        """
        Save scoring results to JSONL file
        
        Args:
            results: List of scoring results
            file_path: Output file path
            append: Whether to use append mode
        """
        mode = 'a' if append else 'w'
        
        with open(file_path, mode, encoding='utf-8') as f:
            for result in results:
                data = {
                    'question_id': result.question_id,
                    'agent_name': result.agent_name,
                    'method': result.method,
                    'criterion_content': result.criterion_content,
                    'criterion_score': result.criterion_score,
                    'satisfied': result.satisfied,
                    'reasoning': result.reasoning
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(results)} scoring results to {file_path}")
    
    @staticmethod
    def load_existing_results(file_path: str) -> set:
        """
        Load existing scoring results for resumption
        
        Args:
            file_path: Results file path
            
        Returns:
            Set of completed (question_id, agent_name, criterion_content) tuples
        """
        completed = set()
        
        if not os.path.exists(file_path):
            return completed
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        key = (
                            data.get('question_id', ''),
                            data.get('agent_name', ''),
                            data.get('criterion_content', '')
                        )
                        completed.add(key)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(completed)} completed scoring results from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
        
        return completed


class AttachmentLoader:
    """Local Attachment Loader"""
    
    def __init__(self, base_path: str = "Attachments"):
        """
        Initialize attachment loader
        
        Args:
            base_path: Base path for attachments
        """
        self.base_path = Path(base_path)
    
    def get_question_attachment_path(self, filename: str) -> Optional[str]:
        """
        Get full path for question attachment
        
        Args:
            filename: Attachment filename
            
        Returns:
            Full path or None
        """
        path = self.base_path / "Questions" / filename
        if path.exists():
            return str(path)
        
        # Try filename conversion (upstream system may convert . to _)
        converted_filename = self._convert_filename(filename)
        if converted_filename != filename:
            converted_path = self.base_path / "Questions" / converted_filename
            if converted_path.exists():
                return str(converted_path)
        
        return None
    
    def get_answer_attachment_path(self, agent_name: str, filename: str) -> Optional[str]:
        """
        Get full path for answer attachment
        
        Args:
            agent_name: Agent name
            filename: Attachment filename
            
        Returns:
            Full path or None
        """
        path = self.base_path / agent_name / filename
        if path.exists():
            return str(path)
        
        # Try filename conversion
        converted_filename = self._convert_filename(filename)
        if converted_filename != filename:
            converted_path = self.base_path / agent_name / converted_filename
            if converted_path.exists():
                return str(converted_path)
        
        return None
    
    def get_reference_answer_attachment_path(self, filename: str) -> Optional[str]:
        """
        Get full path for reference answer attachment
        
        Args:
            filename: Attachment filename
            
        Returns:
            Full path or None
        """
        path = self.base_path / "Reference_answer" / filename
        if path.exists():
            return str(path)
        
        # Try filename conversion (upstream system may convert . to _)
        converted_filename = self._convert_filename(filename)
        if converted_filename != filename:
            converted_path = self.base_path / "Reference_answer" / converted_filename
            if converted_path.exists():
                return str(converted_path)
        
        return None
    
    def _convert_filename(self, filename: str) -> str:
        """
        Convert filename to upstream system format
        Convert all . to _ except for the extension
        """
        if not filename or '.' not in filename:
            return filename
        
        parts = filename.rsplit('.', 1)
        
        if len(parts) == 2 and parts[1] and '.' not in parts[1]:
            name_part, extension = parts
            converted_name = name_part.replace('.', '_')
            return f"{converted_name}.{extension}"
        else:
            return filename.replace('.', '_')
    
    def list_question_attachments(self) -> List[str]:
        """List all question attachments"""
        folder = self.base_path / "Questions"
        if not folder.exists():
            return []
        return [f.name for f in folder.iterdir() if f.is_file()]
    
    def list_answer_attachments(self, agent_name: str) -> List[str]:
        """List all answer attachments for a specific agent"""
        folder = self.base_path / agent_name
        if not folder.exists():
            return []
        return [f.name for f in folder.iterdir() if f.is_file()]
