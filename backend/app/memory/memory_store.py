import json
import os
from typing import Dict, Any
from threading import Lock

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory_data.json")

memory_db: Dict[str, Dict[str, Any]] = {}
memory_lock = Lock()

MAX_MISTAKES = 10
MAX_RECENT_SCORES = 10
ALLOWED_DIFFICULTY = {"beginner", "interview"}
SUPPORTED_LANGUAGES = {"c", "cpp", "python", "java", "csharp"}


def _load_from_disk() -> None:
    global memory_db
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memory_db = json.load(f)
        except Exception:
            memory_db = {}


def _save_to_disk() -> None:
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_db, f, indent=2)
    except Exception as e:
        print(f"[MEMORY] Failed to persist to disk: {e}")


def _init_user(user_id: str) -> None:
    """
    Initialises memory for a user.
    Schema:
    {
        "by_language": {
            "c":      { "common_mistakes": [], "recent_scores": [] },
            "cpp":    { ... },
            "python": { ... },
            "java":   { ... },
            "csharp": { ... }
        },
        "difficulty_level": "beginner"
    }
    Must be called with memory_lock already held.
    """
    if user_id not in memory_db:
        memory_db[user_id] = {
            "by_language": {lang: {"common_mistakes": [], "recent_scores": []}
                            for lang in SUPPORTED_LANGUAGES},
            "difficulty_level": "beginner"
        }
    else:
        # Migrate old flat schema if needed
        user = memory_db[user_id]
        if "by_language" not in user:
            old_mistakes = user.get("common_mistakes", [])
            old_scores   = user.get("recent_scores", [])
            user["by_language"] = {
                lang: {
                    "common_mistakes": old_mistakes if lang == "c" else [],
                    "recent_scores":   old_scores   if lang == "c" else []
                }
                for lang in SUPPORTED_LANGUAGES
            }
            user.pop("common_mistakes", None)
            user.pop("recent_scores", None)
        # Ensure all languages exist
        for lang in SUPPORTED_LANGUAGES:
            if lang not in user["by_language"]:
                user["by_language"][lang] = {"common_mistakes": [], "recent_scores": []}


def get_user_memory(user_id: str, language: str = "c") -> Dict[str, Any]:
    """Returns language-specific memory for a user."""
    lang = language.strip().lower() if language else "c"
    with memory_lock:
        _init_user(user_id)
        lang_mem = memory_db[user_id]["by_language"].get(lang, {"common_mistakes": [], "recent_scores": []})
        return {
            "common_mistakes":  list(lang_mem["common_mistakes"]),
            "recent_scores":    list(lang_mem["recent_scores"]),
            "difficulty_level": memory_db[user_id].get("difficulty_level", "beginner")
        }


def get_all_language_memory(user_id: str) -> Dict[str, Any]:
    """Returns full by_language dict for a user (used by mistake_fixer)."""
    with memory_lock:
        _init_user(user_id)
        return {
            "by_language":      {k: dict(v) for k, v in memory_db[user_id]["by_language"].items()},
            "difficulty_level": memory_db[user_id].get("difficulty_level", "beginner")
        }


def update_user_memory(user_id: str, new_data: Dict[str, Any], language: str = "c") -> None:
    """Updates language-specific memory for a user."""
    lang = language.strip().lower() if language else "c"
    with memory_lock:
        _init_user(user_id)
        user    = memory_db[user_id]
        lang_mem = user["by_language"].setdefault(lang, {"common_mistakes": [], "recent_scores": []})

        if "common_mistakes" in new_data:
            value = new_data["common_mistakes"]
            if isinstance(value, list):
                combined = lang_mem["common_mistakes"] + value
                seen, deduped = set(), []
                for item in combined:
                    if isinstance(item, str) and item not in seen:
                        seen.add(item)
                        deduped.append(item)
                lang_mem["common_mistakes"] = deduped[-MAX_MISTAKES:]

        if "recent_scores" in new_data:
            value = new_data["recent_scores"]
            if isinstance(value, (int, float)):
                lang_mem["recent_scores"].append(max(0, min(10, float(value))))
                lang_mem["recent_scores"] = lang_mem["recent_scores"][-MAX_RECENT_SCORES:]

        if "difficulty_level" in new_data:
            value = new_data["difficulty_level"]
            if isinstance(value, str) and value in ALLOWED_DIFFICULTY:
                user["difficulty_level"] = value

        _save_to_disk()
        print(f"[MEMORY UPDATE] User: {user_id} | Lang: {lang} | {lang_mem}")


# Load persisted memory on module import
_load_from_disk()
