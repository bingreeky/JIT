"""
Evaluation module for DeepPlanning TravelPlanning benchmark.

Ported faithfully from the original DeepPlanning evaluation pipeline:
  QwenLM/Qwen-Agent/benchmark-deepplanning/travelplanning/evaluation/

Three-stage evaluation:
  1. Format conversion: LLM-based conversion of plan text → structured JSON
  2. Commonsense constraints: 20 checks across 8 equally-weighted dimensions
  3. Hard constraints: Task-specific personalized constraint verification

Scoring:
  - commonsense_weighted_score: weighted dimension scores (one-vote veto per dimension)
  - personalized_score: 1.0 if ALL hard constraints pass, else 0.0
  - composite_score: (commonsense + personalized) / 2
  - case_acc: 1.0 only if both commonsense and personalized are 1.0
"""

from .constraints_commonsense import eval_commonsense, EVALUATION_DIMENSIONS
from .constraints_hard import eval_hard

__all__ = ['eval_commonsense', 'eval_hard', 'EVALUATION_DIMENSIONS']
