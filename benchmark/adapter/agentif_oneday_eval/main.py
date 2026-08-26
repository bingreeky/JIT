#!/usr/bin/env python3
"""
LLM Auto Scoring Script
Modify the configuration below and run: python -m benchmark.adapter.agentif_oneday_eval.main
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

from .config import get_settings
from .logging_config import setup_logging, get_logger
from .data_loader import DataLoader
from .scorer import AsyncScorer, ScoringProgress


# ============================================================
# Configuration Section - Modify parameters here
# ============================================================

# Input file paths
QUESTIONS_FILE = "Data/jsonl_eng/questions_104_1214.jsonl"  # Questions file (JSONL format)
ANSWERS_FILE = "Data/chatgpt_agent_2601_1-agent_results.jsonl"      # Answers file (JSONL format)

# Output file path (timestamp will be added automatically, e.g., score_results_20251210_2217.jsonl)
OUTPUT_FILE_PREFIX = "Data/Res/results"       # Output file prefix

# LLM Model configuration
# Model name format: Use official model names with prefix
# Supported prefixes: gemini-, gpt-, qwen/, openrouter/
# Examples: gemini-3-flash-preview, gpt-4o, openrouter/qwen/qwen3.5-122b-a10b
# Any model name starting with these prefixes is supported (no predefined list)
# Leave empty to use the default model from config file
MODEL_NAME = "qwen/qwen3.5-122b-a10b"

# Concurrency and retry configuration (leave empty to use default values from config file)
MAX_CONCURRENT = None  # Maximum concurrent requests, e.g., 5
MAX_RETRIES = None     # Maximum retry attempts, e.g., 3

# Attachment path configuration
ATTACHMENT_BASE_PATH = "Attachments"

# Question filter configuration
# Set to None or [] to score all questions
# Set to a list of question_ids to score only specific questions, e.g., ["taskif_6", "taskif_7"]
QUESTION_IDS = []

# Agent filter configuration
# Set to None or [] to score all agents' answers
# Set to a list of agent_names to score only specific agents, e.g., ["ChatGPT-Agent-2511", "Claude-Agent"]
AGENT_NAMES = None

# Whether to show verbose logs
VERBOSE = False

# ============================================================
# No modifications needed below this line
# ============================================================


def print_progress(progress: ScoringProgress):
    """Print progress information"""
    percentage = (progress.completed_tasks / progress.total_tasks * 100) if progress.total_tasks > 0 else 0
    print(
        f"\rProgress: {progress.completed_tasks}/{progress.total_tasks} ({percentage:.1f}%) "
        f"| Current: {progress.current_task[:50]}{'...' if len(progress.current_task) > 50 else ''}",
        end="",
        flush=True
    )


def list_available_models():
    """List example models (any model with valid prefix is supported)"""
    print("Example LLM models (any model with valid prefix is supported):")
    print("-" * 40)
    print("Supported prefixes: gemini-, gpt-, qwen/, openrouter/")
    print("\nExamples:")
    print("  - gemini-3-flash-preview")
    print("  - gemini-2.5-pro")
    print("  - gpt-4o")
    print("  - gpt-5.1")
    print("  - openrouter/qwen/qwen3.5-122b-a10b")
    print("  - openrouter/openai/gpt-4o")
    print("\nYou can use any official model name with these prefixes.")


async def main():
    """Main function"""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Adjust log level if verbose is set
    if VERBOSE:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate output filename with timestamp
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M")
    output_file = f"{OUTPUT_FILE_PREFIX}_{timestamp}.jsonl"
    
    # Check if files exist
    if not Path(QUESTIONS_FILE).exists():
        print(f"Error: Questions file does not exist: {QUESTIONS_FILE}")
        return 1
    
    if not Path(ANSWERS_FILE).exists():
        print(f"Error: Answers file does not exist: {ANSWERS_FILE}")
        return 1
    
    # Get settings
    settings = get_settings()
    
    # Determine which model to use
    model_name = MODEL_NAME or settings.llm_default_model
    
    # Validate model name format (must start with known prefix)
    valid_prefixes = ("gemini-", "gpt-", "openrouter/", "qwen/")
    if not any(model_name.startswith(prefix) for prefix in valid_prefixes):
        print(f"Error: Invalid model name format: {model_name}")
        print(f"Model name must start with one of: {', '.join(valid_prefixes)}")
        print(f"Examples: gemini-3-flash-preview, gpt-4o, openrouter/qwen/qwen3.5-122b-a10b")
        return 1
    
    # Determine concurrency and retry count
    max_concurrent = MAX_CONCURRENT or settings.llm_max_concurrent
    max_retries = MAX_RETRIES or settings.llm_max_retries
    
    print("=" * 60)
    print("LLM Auto Scoring Script")
    print("=" * 60)
    print(f"Questions file: {QUESTIONS_FILE}")
    print(f"Answers file: {ANSWERS_FILE}")
    print(f"Output file: {output_file}")
    print(f"Model: {model_name}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Max retries: {max_retries}")
    print(f"Attachment path: {ATTACHMENT_BASE_PATH}")
    
    # Display filter configuration
    if QUESTION_IDS:
        print(f"Question filter: {QUESTION_IDS}")
    else:
        print(f"Question filter: All questions")
    
    if AGENT_NAMES:
        print(f"Agent filter: {AGENT_NAMES}")
    else:
        print(f"Agent filter: All agents")
    
    print("=" * 60)
    print()
    
    try:
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(ATTACHMENT_BASE_PATH)
        questions = data_loader.load_questions(QUESTIONS_FILE)
        answers = data_loader.load_answers(ANSWERS_FILE)
        
        print(f"Raw data: {len(questions)} questions, {len(answers)} answers")
        
        # Filter questions
        if QUESTION_IDS:
            question_id_set = set(QUESTION_IDS)
            questions = [q for q in questions if q.question_id in question_id_set]
            print(f"Filtered questions: {len(questions)}")
        
        # Filter answers
        if AGENT_NAMES:
            agent_name_set = set(AGENT_NAMES)
            answers = [a for a in answers if a.agent_name in agent_name_set]
            print(f"Filtered answers (by agent): {len(answers)}")
        
        # If question filter is specified, also filter answers
        if QUESTION_IDS:
            question_id_set = set(QUESTION_IDS)
            answers = [a for a in answers if a.question_id in question_id_set]
            print(f"Filtered answers (by question): {len(answers)}")
        
        if not questions or not answers:
            print("Error: No valid questions or answers after filtering")
            return 1
        
        # Create scorer
        scorer = AsyncScorer(
            model_name=model_name,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            attachment_base_path=ATTACHMENT_BASE_PATH
        )
        
        # Execute scoring
        print("\nStarting scoring...")
        
        results = await scorer.score_all(
            questions=questions,
            answers=answers,
            output_file=output_file,
            skip_existing=False,  # Do not skip existing results
            progress_callback=print_progress
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n")  # New line (since progress output has no newline)
        print("=" * 60)
        print("Scoring completed!")
        print("=" * 60)
        print(f"Total scored items: {len(results)}")
        print(f"Total time: {duration:.1f} seconds")
        print(f"Results saved to: {output_file}")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nUser interrupted, scoring stopped")
        print(f"Completed scoring results have been saved to: {output_file}")
        return 130
    
    except Exception as e:
        logger.exception(f"Scoring process error: {e}")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
