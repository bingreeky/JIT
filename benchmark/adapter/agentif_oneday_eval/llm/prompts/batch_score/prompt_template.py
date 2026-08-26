"""
LLM Batch Scoring Prompt Template
"""
from datetime import date as _date


def build_batch_score_prompt(
    question_text: str,
    question_attachments: list = None,
    answer_text: str = "",
    answer_attachments: list = None,
    score_criteria: list = None,
    use_grounding: bool = False,
    reference_answer_text: str = "",
    reference_answer_attachments: list = None,
    web_search_facts: str = ""
) -> str:
    """
    Build the prompt for LLM batch scoring
    
    Args:
        question_text: Question content
        question_attachments: List of question attachments [{"filename": "", "content": ""}]
        answer_text: Answer content
        answer_attachments: List of answer attachments [{"filename": "", "content": ""}]
        score_criteria: List of scoring criteria [{"content": "", "score": int}]
        use_grounding: Whether to enable Google Search grounding for fact-checking
        reference_answer_text: Reference answer description
        reference_answer_attachments: List of reference answer attachments [{"filename": "", "content": ""}]
        web_search_facts: Pre-fetched web search facts (used in fallback mode)
    
    Returns:
        str: Complete prompt
    """
    
    prompt_parts = []
    
    # 1. Role definition
    grounding_principle = ""
    if use_grounding:
        grounding_principle = "\n- Use Google Search for fact-checking to ensure scoring accuracy"
    
    today = _date.today().isoformat()

    prompt_parts.append(f"""# LLM Auto Scoring System

## Role
You are a professional scoring expert. You need to objectively and fairly score answers based on question requirements, answer content, and scoring criteria.

## Current Date
Today's date is {today}. Use this when judging whether data, events, or facts referenced in the answer could reasonably exist. Do NOT rely on your own training data cutoff to determine whether information is fabricated — if the answer cites data from a date before today, that data may well be real even if you are not aware of it.

## Scoring Principles
- Score strictly according to the scoring criteria
- Provide a clear pass/fail judgment for each scoring criterion
- Provide clear scoring reasons
- Maintain objectivity and fairness
- When evaluating factual accuracy, consider that the agent had access to live web search and may have retrieved information that is more recent than your training data{grounding_principle}""")
    
    # 2. Question content
    prompt_parts.append(f"""
## Question Content
{question_text}""")
    
    # 3. Question attachments
    if question_attachments:
        prompt_parts.append("\n## Question Attachments")
        for i, attachment in enumerate(question_attachments, 1):
            prompt_parts.append(f"""
### Attachment {i}: {attachment.get('filename', f'Attachment{i}')}
{attachment.get('content', '')}""")
    
    # 4. Answer content
    prompt_parts.append(f"""
## Answer to be Scored
{answer_text}""")
    
    # 5. Answer attachments
    if answer_attachments:
        prompt_parts.append("\n## Answer Attachments")
        for i, attachment in enumerate(answer_attachments, 1):
            prompt_parts.append(f"""
### Attachment {i}: {attachment.get('filename', f'Attachment{i}')}
{attachment.get('content', '')}""")
    
    # 6. Reference answer
    if reference_answer_text or reference_answer_attachments:
        prompt_parts.append("""
## Reference Answer
The reference answer is the standard answer used to help judge whether the answer to be scored is correct. When scoring, compare the answer to be scored with the reference answer.""")
        
        if reference_answer_text:
            prompt_parts.append(f"""
### Reference Answer Description
{reference_answer_text}""")
        
        if reference_answer_attachments:
            prompt_parts.append("\n### Reference Answer Attachments")
            for i, attachment in enumerate(reference_answer_attachments, 1):
                prompt_parts.append(f"""
#### Reference Attachment {i}: {attachment.get('filename', f'Attachment{i}')}
{attachment.get('content', '')}""")
    
    # 7. Web search facts (if provided)
    if web_search_facts:
        prompt_parts.append("""
## Pre-fetched Web Search Facts
The following information was obtained through web search to help verify facts in the question and answer:
""")
        prompt_parts.append(web_search_facts)
    
    # 8. Scoring criteria
    if score_criteria:
        prompt_parts.append("\n## Scoring Criteria")
        for i, criterion in enumerate(score_criteria, 1):
            prompt_parts.append(f"""
### Criterion {i} (ID: {criterion.get('criterion_id', '')}, Score: {criterion.get('score', 0)})
{criterion.get('content', '')}""")
    
    # 9. Output requirements
    prompt_parts.append("""
## Scoring Requirements
Please score the above answer according to each scoring criterion, providing for each criterion:
1. **Judgment Result**: Return True when the answer matches the scoring criterion description (add or deduct points); Return False when it does not match (no points added or deducted)
2. **Scoring Reason**: Explain in detail why this judgment was made

## Output Format
Please strictly follow the JSON format below for scoring results:

```json
{
  "criteria_results": [
    {
      "criterion_id": "UUID of the scoring criterion",
      "satisfied": true,
      "reasoning": "Detailed scoring reason..."
    },
    {
      "criterion_id": "UUID of the scoring criterion", 
      "satisfied": false,
      "reasoning": "Detailed scoring reason..."
    }
  ]
}
```

Notes:
- Strictly use JSON format for output
- The satisfied field can only be true or false
- The reasoning must provide specific and detailed reasons
- Ensure criterion_id corresponds to the scoring criterion ID
- If the scoring criterion includes requirements for answer attachments but the answer has no attachments, return false
- Questions and answers may contain both Chinese and English. Pay attention to language conversion. For example, if the scoring criterion requires the date to be "Sunday" but the answer says "星期日" (Sunday in Chinese), it should still be considered as meeting the requirement
- If a reference answer is provided, compare the answer to be scored with the reference answer to determine if it meets the scoring criteria. The reference answer serves as the standard for correctness
""")
    
    return "\n".join(prompt_parts)
