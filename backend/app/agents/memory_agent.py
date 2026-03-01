import json
import re
from app.services.llm_service import generate_response


def memory_agent(analysis_text: str, score_data: dict) -> dict:
    """
    Extracts recurring mistake categories only if performance is weak.
    Always stores recent score.
    """

    overall_score = score_data.get("overall_score")

    # Always store score if valid
    score_value = overall_score if isinstance(overall_score, (int, float)) else None

    #  If score is high, do NOT extract mistakes
    if isinstance(score_value, (int, float)) and score_value >= 7:
        return {
            "common_mistakes": [],
            "recent_scores": score_value
        }

    # --------------------
    # Only analyze mistakes if score is low
    # --------------------
    prompt = f"""
You are a coding mentor memory analyzer.

From the following analysis, extract REAL mistake categories.

IMPORTANT:
- Only extract mistakes if actual errors exist.
- If analysis says code is correct or has no major issues, return:
    {{
    "common_mistakes": []
    }}

Return STRICT JSON in this format:

{{
    "common_mistakes": ["..."]
}}

Only include high-level mistake types like:
- uninitialized variable
- syntax error
- poor edge case handling
- unclear formatting

Analysis:
{analysis_text}
"""

    response = generate_response(prompt, "llama-3.1-8b-instant", 180)

    # --------------------
    # JSON Fence Cleaning
    # --------------------
    response = response.strip()
    response = re.sub(r"^```json", "", response)
    response = re.sub(r"^```", "", response)
    response = re.sub(r"```$", "", response)
    response = response.strip()

    # --------------------
    # Safe JSON Parsing
    # --------------------
    try:
        data = json.loads(response)
        common_mistakes = data.get("common_mistakes", [])

        if not isinstance(common_mistakes, list):
            common_mistakes = []

        return {
            "common_mistakes": common_mistakes,
            "recent_scores": score_value
        }

    except json.JSONDecodeError:
        return {
            "common_mistakes": [],
            "recent_scores": score_value
        }