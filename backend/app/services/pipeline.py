import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.validators import validate_code_input
from app.utils.injection_guard import detect_prompt_injection

from app.agents.analyzer_agent import analyzer_agent
from app.agents.pedagogy_agent import pedagogy_agent
from app.agents.fix_agent import fix_agent
from app.agents.practice_agent import practice_agent
from app.agents.scoring_agent import scoring_agent
from app.agents.intent_router import intent_router
from app.agents.mistake_fixer_agent import mistake_fixer_agent
from app.agents.memory_agent import memory_agent

from app.memory.memory_store import update_user_memory


MAX_CODE_LENGTH = 5000


def _safe_run(fn, *args):
    """Run an agent function, returning an error dict if it raises."""
    try:
        return fn(*args)
    except Exception as e:
        return {"error": f"{fn.__name__} failed: {str(e)}"}


def run_pipeline(
    code: str,
    language: str = "c",
    mode: str = "beginner",
    user_query: str = "full review",
    user_id: str = "default_user",
    intent: str | None = None
):
    # ----------------------------
    # Mistake Fixer — memory-only, skip code validation
    # ----------------------------
    if intent and intent.strip().lower() == "mistake_fixer":
        language = (language or "c").strip().lower()
        return {"mistake_fixer": _safe_run(mistake_fixer_agent, user_id, language)}

    # ----------------------------
    # Validation
    # ----------------------------
    if not validate_code_input(code):
        return {"error": "Invalid or empty code input."}

    if len(code) > MAX_CODE_LENGTH:
        return {"error": "Code exceeds maximum allowed length."}

    if detect_prompt_injection(code) or detect_prompt_injection(user_query):
        return {"error": "Potential prompt injection detected."}

    # Normalize language
    language = (language or "c").strip().lower()

    # ----------------------------
    # Intent Routing
    # ----------------------------
    if intent:
        intent = intent.strip().lower()
    else:
        query_lower = user_query.lower()
        if any(w in query_lower for w in ["score", "rate", "evaluate", "grade"]):
            intent = "score"
        elif any(w in query_lower for w in ["fix", "correct", "debug"]):
            intent = "fix"
        elif "explain" in query_lower:
            intent = "explain"
        elif "practice" in query_lower:
            intent = "practice"
        elif any(w in query_lower for w in ["mistake", "common error", "fix my habit", "weak"]):
            intent = "mistake_fixer"
        elif any(w in query_lower for w in ["analyze", "analysis"]):
            intent = "analyze"
        else:
            intent_data = intent_router(code, user_query)
            intent = intent_data.get("intent", "full_review").strip().lower()

    print(f"Final Intent: {intent}  Language: {language}  Mode: {mode}")

    response = {}

    # ----------------------------
    # Single Agent Execution
    # ----------------------------
    if intent == "analyze":
        response["analysis"] = _safe_run(analyzer_agent, code, language)

    elif intent == "explain":
        response["explanation"] = _safe_run(pedagogy_agent, code, mode, language)

    elif intent == "fix":
        fix_raw = _safe_run(fix_agent, code, language)
        try:
            response["fix"] = json.loads(fix_raw) if isinstance(fix_raw, str) else fix_raw
        except json.JSONDecodeError:
            response["fix"] = {"error": "Invalid JSON from fix agent.", "raw_response": fix_raw}

    elif intent == "practice":
        practice_result = _safe_run(practice_agent, code, mode, user_id, language)
        if isinstance(practice_result, dict) and "error" not in practice_result:
            response["practice"] = {
                "similar_problems": practice_result.get("similar_problems", []),
                "challenge_problem": practice_result.get("challenge_problem", ""),
                "hints": practice_result.get("hints", [])
            }
        else:
            response["practice"] = practice_result

    elif intent == "score":
        response["score"] = _safe_run(scoring_agent, code, language)

    elif intent == "mistake_fixer":
        response["mistake_fixer"] = _safe_run(mistake_fixer_agent, user_id, language)

    # ----------------------------
    # Full Review
    # ----------------------------
    else:
        # Run all agents in parallel except scoring (needs analysis first)
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_analysis    = ex.submit(_safe_run, analyzer_agent,  code, language)
            fut_explanation = ex.submit(_safe_run, pedagogy_agent,  code, mode, language)
            fut_fix         = ex.submit(_safe_run, fix_agent,       code, language)
            fut_practice    = ex.submit(_safe_run, practice_agent,  code, mode, user_id, language)
        analysis        = fut_analysis.result()
        explanation     = fut_explanation.result()
        fix_raw         = fut_fix.result()
        practice_result = fut_practice.result()
        # Score last — benefits from analysis context
        score           = _safe_run(scoring_agent, code, language, analysis)

        try:
            fix_data = json.loads(fix_raw) if isinstance(fix_raw, str) else fix_raw
        except (json.JSONDecodeError, TypeError):
            fix_data = {"error": "Invalid JSON from fix agent.", "raw_response": fix_raw}

        if isinstance(practice_result, dict) and "error" not in practice_result:
            practice_data = {
                "similar_problems": practice_result.get("similar_problems", []),
                "challenge_problem": practice_result.get("challenge_problem", ""),
                "hints": practice_result.get("hints", [])
            }
        else:
            practice_data = practice_result

        response = {
            "analysis":    analysis,
            "explanation": explanation,
            "fix":         fix_data,
            "practice":    practice_data,
            "score":       score,
        }

    # ----------------------------
    # Memory Update — runs after every intent
    # ----------------------------
    analysis_data = response.get("analysis", {})
    score_data    = response.get("score", {})

    # If this intent didnt produce analysis or score, run both silently for memory
    if not analysis_data and not score_data:
        try:
            analysis_data = analyzer_agent(code, language)
        except Exception:
            analysis_data = {}
        try:
            score_data = scoring_agent(code, language, analysis_data)
        except Exception:
            score_data = {}

    # If only one of them exists, fill in the other
    elif analysis_data and not score_data:
        try:
            score_data = scoring_agent(code, language, analysis_data)
        except Exception:
            score_data = {}
    elif score_data and not analysis_data:
        try:
            analysis_data = analyzer_agent(code, language)
        except Exception:
            analysis_data = {}

    try:
        memory_update = memory_agent(str(analysis_data), score_data)
        update_user_memory(user_id, memory_update, language)
    except Exception:
        pass

    return response