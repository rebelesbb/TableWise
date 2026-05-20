from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent

SAMPLE_PATH = ROOT / "data" / "app_restaurants_sample.csv"
FULL_DATA_PATH = ROOT / "data" / "data_new" / "processed" / "restaurants_processed.csv"
DATA_PATH = SAMPLE_PATH if SAMPLE_PATH.exists() else FULL_DATA_PATH
FEEDBACK_PATH = ROOT / "data" / "artifacts" / "feedback" / "ui_feedback.csv"

ARTIFACTS_DIR = ROOT / "data" / "artifacts_new"
if not ARTIFACTS_DIR.exists() and (ROOT / "artifacts_new").exists():
    ARTIFACTS_DIR = ROOT / "artifacts_new"
if not ARTIFACTS_DIR.exists() and (ROOT / "artifacts").exists():
    ARTIFACTS_DIR = ROOT / "artifacts"

FAISS_DIR = ARTIFACTS_DIR / "faiss"
MAPPING_PATH = FAISS_DIR / "restaurant_index_mapping.parquet"
MAPPING_CSV_PATH = FAISS_DIR / "restaurant_index_mapping.csv"
EMBEDDINGS_PATH = FAISS_DIR / "restaurant_embeddings.npy"
FAISS_INDEX_PATH = FAISS_DIR / "restaurant_faiss.index"
FAISS_METADATA_PATH = FAISS_DIR / "metadata.json"

SLM_ADAPTER_PATHS = [
    ARTIFACTS_DIR / "slm_query_parser" / "qwen2_5_1_5b_query_parser_lora",
    ARTIFACTS_DIR / "slm_query_parser_2" / "qwen2_5_1_5b_query_parser_lora",
    ROOT / "data" / "artifacts_new" / "slm_query_parser" / "qwen2_5_1_5b_query_parser_lora",
    ROOT / "data" / "artifacts_new" / "slm_query_parser_2" / "qwen2_5_1_5b_query_parser_lora",
]
SLM_ADAPTER_PATH = next((path for path in SLM_ADAPTER_PATHS if path.exists()), SLM_ADAPTER_PATHS[0])

REWARD_MODEL_PATH = ARTIFACTS_DIR / "feedback_rlhf" / "reward_model" / "feedback_reward_model.joblib"
REWARD_FEATURES_PATH = ARTIFACTS_DIR / "feedback_rlhf" / "reward_model" / "reward_feature_columns.json"

DEFAULT_LIMIT = 9
SLM_MAX_NEW_TOKENS = 160
RAG_MAX_NEW_TOKENS = 280

PRICE_WORDS = {
    "cheap": {"cheap", "budget", "ieftin", "ieftina", "ieftine", "low cost", "affordable"},
    "mid": {"mid", "moderate", "mediu", "mijloc", "mid-range", "casual"},
    "expensive": {"expensive", "premium", "fine dining", "luxury", "scump", "scumpa", "scumpe"},
}

MEAL_WORDS = {
    "breakfast": {"breakfast", "mic dejun", "coffee"},
    "brunch": {"brunch"},
    "lunch": {"lunch", "pranz", "pranzul"},
    "dinner": {"dinner", "cina", "evening", "seara"},
    "drinks": {"drinks", "bar", "cocktail", "wine"},
}

COMMON_TAGS = {
    "italian", "pizza", "french", "spanish", "greek", "portuguese", "german", "seafood", "steakhouse",
    "vegetarian", "vegetarian friendly", "vegan", "vegan options", "gluten free", "gluten free options",
    "sushi", "asian", "indian", "mexican", "mediterranean", "fast food", "bar", "tapas", "cafe",
    "coffee", "romantic", "family", "family friendly", "outdoor seating", "reservations", "delivery",
    "wheelchair accessible", "fine dining", "cheap eats", "healthy",
}

REWARD_FEATURE_COLUMNS = [
    "semantic_score_norm",
    "metadata_score_norm",
    "rating_score",
    "popularity_score_norm",
    "soft_constraint_score",
    "hard_constraint_score",
]

RERANK_WEIGHTS = {
    "semantic": 0.40,
    "metadata": 0.20,
    "rating": 0.15,
    "popularity": 0.10,
    "soft_constraints": 0.15,
}

FEEDBACK_RERANK_WEIGHTS = {
    "original_rerank": 0.70,
    "reward_score": 0.30,
}


@dataclass
class ParsedQuery:
    city: str = ""
    country: str = ""
    price_bucket: str = ""
    meal: str = ""
    tags: tuple[str, ...] = ()
    exclude_tags: tuple[str, ...] = ()
    min_rating: float = 0.0


_COMPONENTS: dict[str, object] = {}
_COMPONENT_STATUS: dict[str, object] = {
    "slm_parser": "not loaded",
    "faiss": "not loaded",
    "embedding_model": "not loaded",
    "reward_model": "not loaded",
    "rag_generator": "not loaded",
}


def log_step(step: str, message: str = "") -> None:
    """Print pipeline progress in the terminal.

    Disable with TABLEWISE_VERBOSE=0 if the output becomes too noisy.
    """
    if os.environ.get("TABLEWISE_VERBOSE", "1").strip().lower() in {"0", "false", "no"}:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    suffix = f" - {message}" if message else ""
    print(f"[{timestamp}] [TableWise] {step}{suffix}", flush=True)


def clean(value: object | None) -> str:
    return ("" if value is None else str(value)).strip()


def norm(value: object | None) -> str:
    return clean(value).lower()


def safe_float(value: object | None, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan"):
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def normalize_price(value: object) -> str:
    value = norm(value)
    if value in {"mid-range", "moderate", "medium"}:
        return "mid"
    return value


def contains_any(text: str, needles: set[str]) -> bool:
    return any(needle in text for needle in needles)


def extract_excluded_tags(query: str) -> tuple[str, ...]:
    q = norm(query)
    excluded: list[str] = []
    for tag in COMMON_TAGS:
        pattern = rf"\b(?:don't want|dont want|do not want|no|not|without|exclude|excluding)\s+(?:any\s+)?{re.escape(tag)}\b"
        if re.search(pattern, q):
            excluded.append(tag)
    return tuple(sorted(set(excluded)))


def parse_query_rules(query: str) -> ParsedQuery:
    q = norm(query)
    price = ""
    for bucket, words in PRICE_WORDS.items():
        if contains_any(q, words):
            price = bucket
            break

    meal = ""
    for meal_name, words in MEAL_WORDS.items():
        if contains_any(q, words):
            meal = meal_name
            break

    exclude_tags = extract_excluded_tags(query)
    tags = tuple(tag for tag in COMMON_TAGS if tag in q and tag not in exclude_tags)

    min_rating = 0.0
    rating_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:star|rating|rated|restaurant)", query, re.IGNORECASE)
    if rating_match:
        min_rating = safe_float(rating_match.group(1))

    city = ""
    city_match = re.search(r"\b(?:in|din|la|of|near)\s+(?:the\s+center\s+of\s+)?([a-zA-Z' -]{3,40})", query, re.IGNORECASE)
    if not city_match:
        city_match = re.search(
            r"\b(?:visiting|visit|going to|traveling to|travelling to|staying in|headed to|heading to|vacation in|weekend in)\s+(?:the\s+center\s+of\s+)?([a-zA-Z' -]{3,40}?)(?=\s+(?:this|next|for|on|with|near|in|at|and|or|to|from|the|weekend|week|month|year)\b|[.?,!]|$)",
            query,
            re.IGNORECASE,
        )
    if city_match:
        city = city_match.group(1).strip(" .,!?")

    return ParsedQuery(city=city, price_bucket=price, meal=meal, tags=tags, exclude_tags=exclude_tags, min_rating=min_rating)


def extract_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def normalize_slm_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({norm(item) for item in value if norm(item)}))


def parsed_from_slm_payload(payload: dict[str, object], fallback: ParsedQuery) -> ParsedQuery:
    price = normalize_price(payload.get("price_bucket") or "") or fallback.price_bucket
    meals = normalize_slm_list(payload.get("matched_meals"))
    tags = normalize_slm_list(payload.get("tags"))
    dietary = normalize_slm_list(payload.get("dietary"))
    features = normalize_slm_list(payload.get("matched_features"))
    meal = next((item for item in meals if item in MEAL_WORDS), fallback.meal)
    clean_tags = tuple(tag for tag in (tags + dietary + features) if tag in COMMON_TAGS)
    city = clean(payload.get("city") or "") or fallback.city
    country = clean(payload.get("country") or "") or fallback.country
    return ParsedQuery(
        city=city if city.lower() != "none" else fallback.city,
        country=country if country.lower() != "none" else fallback.country,
        price_bucket=price,
        meal=meal,
        tags=clean_tags or fallback.tags,
        exclude_tags=fallback.exclude_tags,
        min_rating=fallback.min_rating,
    )


def load_slm_parser() -> bool:
    log_step("SLM parser", "checking/loading fine-tuned Qwen LoRA adapter")
    if "slm_model" in _COMPONENTS and "slm_tokenizer" in _COMPONENTS:
        return True
    if os.environ.get("TABLEWISE_DISABLE_SLM", "").strip().lower() in {"1", "true", "yes"}:
        _COMPONENT_STATUS["slm_parser"] = "disabled"
        log_step("SLM parser", "disabled by TABLEWISE_DISABLE_SLM")
        return False
    if not SLM_ADAPTER_PATH.exists():
        _COMPONENT_STATUS["slm_parser"] = f"adapter not found: {SLM_ADAPTER_PATH}"
        log_step("SLM parser", f"adapter not found: {SLM_ADAPTER_PATH}")
        return False
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        with (SLM_ADAPTER_PATH / "adapter_config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)
        base_model_name = config.get("base_model_name_or_path") or "Qwen/Qwen2.5-1.5B-Instruct"
        local_only = False
        use_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        dtype = torch.float16 if use_cuda else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(SLM_ADAPTER_PATH, local_files_only=local_only, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"local_files_only": local_only, "trust_remote_code": True, "torch_dtype": dtype}
        if use_cuda:
            kwargs["device_map"] = "auto"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        model = PeftModel.from_pretrained(base_model, SLM_ADAPTER_PATH, local_files_only=local_only)
        model.eval()
        if not use_cuda:
            model.to("cpu")
        _COMPONENTS["slm_model"] = model
        _COMPONENTS["slm_tokenizer"] = tokenizer
        _COMPONENT_STATUS["slm_parser"] = f"loaded: {base_model_name} + LoRA adapter"
        log_step("SLM parser", f"loaded {base_model_name} + LoRA adapter")
        return True
    except Exception as error:
        _COMPONENT_STATUS["slm_parser"] = f"unavailable: {error}"
        log_step("SLM parser", f"unavailable: {error}")
        return False


def parse_query_with_slm(query: str, fallback: ParsedQuery) -> ParsedQuery | None:
    log_step("Parse", f"received query: {query}")
    if not query or not load_slm_parser():
        return None
    try:
        import torch
        model = _COMPONENTS["slm_model"]
        tokenizer = _COMPONENTS["slm_tokenizer"]
        schema = {
            "city": "string or null",
            "country": "string or null",
            "price_bucket": "cheap, mid, expensive, or null",
            "tags": "array of cuisine/style tags",
            "dietary": "array of dietary constraints",
            "matched_meals": "array of meals",
            "matched_features": "array of restaurant features",
        }
        messages = [
            {"role": "system", "content": "Return only one valid JSON object for the restaurant query. No markdown, no explanation."},
            {"role": "user", "content": f"Schema: {json.dumps(schema)}\nRestaurant search query: {query.strip().lower()}"},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Parse this restaurant search query to JSON. Query: {query.strip().lower()}\nJSON:"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        log_step("Parse", "generating structured JSON with fine-tuned SLM")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=SLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        payload = extract_json_object(generated)
        if not payload:
            _COMPONENT_STATUS["slm_parser"] = "loaded, but last output was not valid JSON; used rule fallback"
            log_step("Parse", "SLM returned invalid JSON; falling back to rules if allowed")
            return None
        log_step("Parse", f"SLM JSON: {json.dumps(payload, ensure_ascii=False)}")
        return parsed_from_slm_payload(payload, fallback)
    except Exception as error:
        _COMPONENT_STATUS["slm_parser"] = f"loaded, but parsing failed: {error}; used rule fallback"
        return None


def parse_query(query: str) -> tuple[ParsedQuery, str]:
    fallback = parse_query_rules(query)
    parsed = parse_query_with_slm(query, fallback)
    if parsed is not None:
        log_step("Parse", f"using fine-tuned SLM parser -> {parsed}")
        return parsed, "fine-tuned SLM parser"
    if os.environ.get("TABLEWISE_REQUIRE_FULL_PIPELINE", "1").strip().lower() in {"1", "true", "yes"} and query and not load_slm_parser():
        if os.environ.get("TABLEWISE_ALLOW_RULE_FALLBACK", "").strip().lower() not in {"1", "true", "yes"}:
            raise RuntimeError(f"Fine-tuned SLM parser could not be loaded: {_COMPONENT_STATUS['slm_parser']}")
    log_step("Parse", f"using rule-based fallback parser -> {fallback}")
    return fallback, "rule-based fallback parser"


def load_full_retrieval_components() -> bool:
    log_step("FAISS", "checking/loading mapping, embeddings, FAISS index and embedding model")
    if all(key in _COMPONENTS for key in ["mapping_df", "embeddings", "faiss_index", "embedding_model"]):
        return True
    missing = [str(path) for path in [EMBEDDINGS_PATH, FAISS_INDEX_PATH] if not path.exists()]
    if not MAPPING_PATH.exists() and not MAPPING_CSV_PATH.exists():
        missing.append(str(MAPPING_PATH))
    if missing:
        _COMPONENT_STATUS["faiss"] = "missing artifacts: " + "; ".join(missing)
        log_step("FAISS", "missing artifacts: " + "; ".join(missing))
        return False
    try:
        import faiss
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        log_step("FAISS", f"reading mapping from {MAPPING_PATH if MAPPING_PATH.exists() else MAPPING_CSV_PATH}")
        if MAPPING_PATH.exists():
            mapping_df = pd.read_parquet(MAPPING_PATH)
        else:
            mapping_df = pd.read_csv(MAPPING_CSV_PATH)
        log_step("FAISS", f"loading embeddings from {EMBEDDINGS_PATH}")
        embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        log_step("FAISS", f"reading FAISS index from {FAISS_INDEX_PATH}")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        if len(mapping_df) != embeddings.shape[0] or len(mapping_df) != faiss_index.ntotal:
            raise RuntimeError("FAISS index, embeddings, and mapping sizes do not match.")
        metadata = {}
        if FAISS_METADATA_PATH.exists():
            with FAISS_METADATA_PATH.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        model_name = metadata.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
        log_step("FAISS", f"loading embedding model: {model_name}")
        embedding_model = SentenceTransformer(model_name)
        _COMPONENTS.update({
            "mapping_df": mapping_df,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
            "embedding_model": embedding_model,
        })
        _COMPONENT_STATUS["faiss"] = f"loaded: {faiss_index.ntotal:,} vectors"
        _COMPONENT_STATUS["embedding_model"] = f"loaded: {model_name}"
        log_step("FAISS", f"loaded successfully: {faiss_index.ntotal:,} vectors, embeddings shape={embeddings.shape}")
        return True
    except Exception as error:
        _COMPONENT_STATUS["faiss"] = f"unavailable: {error}"
        log_step("FAISS", f"unavailable: {error}")
        return False


def load_reward_model() -> bool:
    log_step("Reward model", "checking/loading Logistic Regression reward model")
    if "reward_model" in _COMPONENTS:
        return True
    if not REWARD_MODEL_PATH.exists():
        _COMPONENT_STATUS["reward_model"] = f"missing: {REWARD_MODEL_PATH}"
        log_step("Reward model", f"missing: {REWARD_MODEL_PATH}")
        return False
    try:
        import joblib
        reward_model = joblib.load(REWARD_MODEL_PATH)
        if REWARD_FEATURES_PATH.exists():
            with REWARD_FEATURES_PATH.open("r", encoding="utf-8") as file:
                features = json.load(file)
            if isinstance(features, list) and features:
                _COMPONENTS["reward_feature_columns"] = features
        _COMPONENTS["reward_model"] = reward_model
        _COMPONENT_STATUS["reward_model"] = "loaded: sklearn Logistic Regression reward model"
        log_step("Reward model", "loaded sklearn Logistic Regression reward model")
        return True
    except Exception as error:
        _COMPONENT_STATUS["reward_model"] = f"unavailable: {error}"
        log_step("Reward model", f"unavailable: {error}")
        return False


def load_rag_generator() -> bool:
    log_step("RAG generator", "checking/loading answer generation model")
    if "rag_model" in _COMPONENTS and "rag_tokenizer" in _COMPONENTS:
        return True
    if os.environ.get("TABLEWISE_DISABLE_RAG_LLM", "").strip().lower() in {"1", "true", "yes"}:
        _COMPONENT_STATUS["rag_generator"] = "disabled; using deterministic template"
        log_step("RAG generator", "disabled; using deterministic template")
        return False
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = os.environ.get("TABLEWISE_RAG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        local_only = False
        use_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        dtype = torch.float16 if use_cuda else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only, trust_remote_code=True)
        kwargs = {"local_files_only": local_only, "trust_remote_code": True, "torch_dtype": dtype}
        if use_cuda:
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model.eval()
        if not use_cuda:
            model.to("cpu")
        _COMPONENTS["rag_model"] = model
        _COMPONENTS["rag_tokenizer"] = tokenizer
        _COMPONENT_STATUS["rag_generator"] = f"loaded: {model_name}"
        log_step("RAG generator", f"loaded {model_name}")
        return True
    except Exception as error:
        _COMPONENT_STATUS["rag_generator"] = f"unavailable: {error}; using deterministic template"
        log_step("RAG generator", f"unavailable: {error}; using deterministic template")
        return False


def row_text(row: object, column: str) -> str:
    try:
        value = row.get(column, "")
    except AttributeError:
        value = ""
    return norm(value)


def compute_metadata_scores(row: object, parsed: ParsedQuery) -> tuple[float, float, float, list[str]]:
    reasons: list[str] = []
    matches = 0
    total = 0
    hard_ok = 1.0

    city = row_text(row, "city_filled") or row_text(row, "city")
    country = row_text(row, "country")
    price = normalize_price(row_text(row, "price_bucket"))
    meals = row_text(row, "meals_text")
    searchable = " ".join([
        row_text(row, "top_tags_text") or row_text(row, "cuisines_text"),
        row_text(row, "special_diets_text"),
        row_text(row, "features_text"),
        row_text(row, "profile_text"),
    ])

    if parsed.city:
        total += 1
        if norm(parsed.city) in city:
            matches += 1
            reasons.append(f"located in {clean(row.get('city_filled') or row.get('city'))}")
        else:
            hard_ok = 0.0
    if parsed.country:
        total += 1
        if norm(parsed.country) in country:
            matches += 1
            reasons.append(f"in {clean(row.get('country'))}")
        else:
            hard_ok = 0.0
    if parsed.price_bucket:
        total += 1
        if parsed.price_bucket == price:
            matches += 1
            reasons.append(f"{parsed.price_bucket} price range")
        else:
            hard_ok = 0.0
    if parsed.meal:
        total += 1
        if parsed.meal in meals:
            matches += 1
            reasons.append(f"serves {parsed.meal}")

    soft_total = 0
    soft_matches = 0
    for tag in parsed.tags:
        soft_total += 1
        if tag in searchable:
            soft_matches += 1
            reasons.append(f"matches {tag}")
    for excluded in parsed.exclude_tags:
        if excluded in searchable:
            hard_ok = 0.0

    metadata_score = matches / max(total, 1)
    soft_score = soft_matches / max(soft_total, 1) if soft_total else 0.0
    if not reasons:
        reasons.append("strong semantic and metadata profile")
    return metadata_score, soft_score, hard_ok, reasons[:4]


def apply_hard_filters(mapping_df, parsed: ParsedQuery):
    import pandas as pd
    mask = pd.Series(True, index=mapping_df.index)
    city_col = (mapping_df.get("city_filled", mapping_df.get("city", "")).fillna("").astype(str).str.lower())
    country_col = mapping_df.get("country", "").fillna("").astype(str).str.lower()
    price_col = mapping_df.get("price_bucket", "").fillna("").astype(str).str.lower().replace({"mid-range": "mid", "moderate": "mid", "medium": "mid"})
    meals_col = mapping_df.get("meals_text", "").fillna("").astype(str).str.lower()
    rating_col = mapping_df.get("rating", 0)

    if parsed.city:
        mask &= city_col.str.contains(re.escape(norm(parsed.city)), na=False)
    if parsed.country:
        mask &= country_col.str.contains(re.escape(norm(parsed.country)), na=False)
    if parsed.price_bucket:
        mask &= price_col.eq(parsed.price_bucket)
    if parsed.meal:
        mask &= meals_col.str.contains(re.escape(parsed.meal), na=False)
    if parsed.min_rating:
        mask &= pd.to_numeric(rating_col, errors="coerce").fillna(0) >= parsed.min_rating
    if parsed.exclude_tags:
        searchable = (
            mapping_df.get("top_tags_text", mapping_df.get("cuisines_text", "")).fillna("").astype(str).str.lower()
            + " " + mapping_df.get("special_diets_text", "").fillna("").astype(str).str.lower()
            + " " + mapping_df.get("features_text", "").fillna("").astype(str).str.lower()
            + " " + mapping_df.get("profile_text", "").fillna("").astype(str).str.lower()
        )
        for tag in parsed.exclude_tags:
            mask &= ~searchable.str.contains(re.escape(tag), na=False)
    return mapping_df[mask].copy()


def minmax(values):
    import numpy as np
    arr = np.asarray(values, dtype="float32")
    if arr.size == 0:
        return arr
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    return (arr - lo) / (hi - lo)


def run_full_pipeline(query: str, limit: int = DEFAULT_LIMIT) -> dict[str, object]:
    total_start = time.perf_counter()
    log_step("PIPELINE START", f"query='{query}', limit={limit}")
    log_step("STEP 1/9", "load FAISS retrieval components")
    if not load_full_retrieval_components():
        raise RuntimeError(f"Full FAISS pipeline could not be loaded: {_COMPONENT_STATUS['faiss']}")
    log_step("STEP 2/9", "parse natural-language query")
    parsed, parser_name = parse_query(query)
    log_step("STEP 2/9", f"parser={parser_name}; parsed={parsed}")

    import numpy as np
    import pandas as pd

    mapping_df = _COMPONENTS["mapping_df"]
    embeddings = _COMPONENTS["embeddings"]
    embedding_model = _COMPONENTS["embedding_model"]

    log_step("STEP 3/9", "apply hard filters: city/country/price/meal/rating/excluded tags")
    filter_start = time.perf_counter()
    candidates = apply_hard_filters(mapping_df, parsed)
    log_step("STEP 3/9", f"candidates after filters: {len(candidates):,} / {len(mapping_df):,} in {time.perf_counter() - filter_start:.2f}s")
    relaxed = False
    if candidates.empty and (parsed.price_bucket or parsed.meal):
        relaxed = True
        relaxed_parsed = ParsedQuery(
            city=parsed.city,
            country=parsed.country,
            tags=parsed.tags,
            exclude_tags=parsed.exclude_tags,
            min_rating=parsed.min_rating,
        )
        log_step("STEP 3/9", "no candidates; relaxing softer price/meal filters")
        candidates = apply_hard_filters(mapping_df, relaxed_parsed)
        log_step("STEP 3/9", f"candidates after relaxed filters: {len(candidates):,}")
    if candidates.empty:
        return {
            "query": query,
            "parsed": parsed.__dict__,
            "parser": parser_name,
            "scanned": len(mapping_df),
            "filtered": 0,
            "results": [],
            "answer": "I could not find a confident match for those constraints in the indexed dataset.",
            "relaxed": relaxed,
            "pipeline": component_status(),
        }

    log_step("STEP 4/9", "encode query with all-MiniLM embedding model")
    embed_start = time.perf_counter()
    query_embedding = embedding_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
    log_step("STEP 4/9", f"query embedding ready in {time.perf_counter() - embed_start:.2f}s")
    faiss_indices = candidates["faiss_idx"].to_numpy(dtype="int64") if "faiss_idx" in candidates.columns else candidates.index.to_numpy(dtype="int64")
    log_step("STEP 5/9", f"compute semantic similarity for {len(faiss_indices):,} filtered candidates")
    sem_start = time.perf_counter()
    semantic_scores = np.empty(len(faiss_indices), dtype="float32")
    chunk_size = 50_000
    for start in range(0, len(faiss_indices), chunk_size):
        idx_chunk = faiss_indices[start:start + chunk_size]
        semantic_scores[start:start + chunk_size] = embeddings[idx_chunk] @ query_embedding

    log_step("STEP 5/9", f"semantic scoring done in {time.perf_counter() - sem_start:.2f}s")
    candidates = candidates.reset_index(drop=True)
    candidates["semantic_score"] = semantic_scores
    candidates["semantic_score_norm"] = minmax(semantic_scores)

    log_step("STEP 6/9", "compute metadata/soft/hard constraint scores")
    meta_start = time.perf_counter()
    metadata_scores = []
    soft_scores = []
    hard_scores = []
    reasons_list = []
    for _, row in candidates.iterrows():
        metadata, soft, hard, reasons = compute_metadata_scores(row, parsed)
        metadata_scores.append(metadata)
        soft_scores.append(soft)
        hard_scores.append(hard)
        reasons_list.append(reasons)
    candidates["metadata_score_norm"] = metadata_scores
    candidates["soft_constraint_score"] = soft_scores
    candidates["hard_constraint_score"] = hard_scores
    candidates["match_reasons"] = reasons_list
    log_step("STEP 6/9", f"metadata scoring done in {time.perf_counter() - meta_start:.2f}s")
    candidates["rating_score"] = pd.to_numeric(candidates.get("rating", 0), errors="coerce").fillna(0).clip(0, 5) / 5.0
    candidates["popularity_score_norm"] = minmax(pd.to_numeric(candidates.get("popularity_score", 0), errors="coerce").fillna(0).to_numpy())

    log_step("STEP 7/9", "compute hybrid rerank score")
    candidates["final_rerank_score"] = candidates["hard_constraint_score"] * (
        RERANK_WEIGHTS["semantic"] * candidates["semantic_score_norm"]
        + RERANK_WEIGHTS["metadata"] * candidates["metadata_score_norm"]
        + RERANK_WEIGHTS["rating"] * candidates["rating_score"]
        + RERANK_WEIGHTS["popularity"] * candidates["popularity_score_norm"]
        + RERANK_WEIGHTS["soft_constraints"] * candidates["soft_constraint_score"]
    )

    log_step("STEP 8/9", "apply RLHF-like reward model if available")
    if load_reward_model():
        reward_model = _COMPONENTS["reward_model"]
        feature_cols = _COMPONENTS.get("reward_feature_columns", REWARD_FEATURE_COLUMNS)
        for col in feature_cols:
            if col not in candidates.columns:
                candidates[col] = 0.0
        candidates["reward_score"] = reward_model.predict_proba(candidates[list(feature_cols)].fillna(0.0))[:, 1]
        candidates["feedback_rerank_score"] = (
            FEEDBACK_RERANK_WEIGHTS["original_rerank"] * candidates["final_rerank_score"]
            + FEEDBACK_RERANK_WEIGHTS["reward_score"] * candidates["reward_score"]
        )
        log_step("STEP 8/9", "reward scores computed and blended with rerank scores")
    else:
        candidates["reward_score"] = 0.0
        candidates["feedback_rerank_score"] = candidates["final_rerank_score"]
        log_step("STEP 8/9", "reward model not available; using hybrid rerank only")

    log_step("STEP 9/9", "select top results and generate grounded answer")
    top_df = candidates.sort_values("feedback_rerank_score", ascending=False).head(limit).copy()
    results: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        results.append({
            "rank": rank,
            "score": round(float(row.get("feedback_rerank_score", 0.0)), 4),
            "semantic_score": round(float(row.get("semantic_score_norm", 0.0)), 4),
            "reward_score": round(float(row.get("reward_score", 0.0)), 4),
            "name": clean(row.get("name")) or "Unknown restaurant",
            "city": clean(row.get("city_filled") or row.get("city")),
            "country": clean(row.get("country")),
            "address": clean(row.get("address")),
            "latitude": clean(row.get("latitude")),
            "longitude": clean(row.get("longitude")),
            "rating": clean(row.get("rating")) or "Unknown",
            "price": clean(row.get("price_bucket")) or "Unknown",
            "tags": clean(row.get("top_tags_text") or row.get("cuisines_text")) or "Unknown",
            "meals": clean(row.get("meals_text")) or "Unknown",
            "features": clean(row.get("features_text")) or "Unknown",
            "popularity": clean(row.get("popularity_detailed")) or "Unknown",
            "profile": clean(row.get("short_profile") or row.get("profile_text")),
            "reasons": row.get("match_reasons") or ["retrieved and reranked by the full pipeline"],
        })

    if results:
        log_step("STEP 9/9", "top results: " + " | ".join([item["name"] for item in results[:3]]))
    answer, answer_method, grounded = generate_grounded_answer(query, results, parsed, relaxed)
    log_step("PIPELINE END", f"answer_method={answer_method}; grounded={grounded}; total_time={time.perf_counter() - total_start:.2f}s")
    return {
        "query": query,
        "parsed": parsed.__dict__,
        "parser": parser_name,
        "scanned": len(mapping_df),
        "filtered": int(len(candidates)),
        "results": results,
        "answer": answer,
        "answer_method": answer_method,
        "grounded": grounded,
        "relaxed": relaxed,
        "pipeline": component_status(),
    }


def build_rag_context(results: list[dict[str, object]]) -> str:
    blocks = []
    for item in results[:5]:
        blocks.append(
            f"Restaurant: {item['name']}\n"
            f"Location: {item['city']}, {item['country']}\n"
            f"Rating: {item['rating']}\n"
            f"Price: {item['price']}\n"
            f"Cuisine/tags: {item['tags']}\n"
            f"Meals: {item['meals']}\n"
            f"Reasons: {', '.join(item.get('reasons', []))}"
        )
    return "\n\n".join(blocks)


def template_answer(query: str, results: list[dict[str, object]], parsed: ParsedQuery, relaxed: bool = False) -> str:
    if not results:
        return "I could not find a confident match for those constraints in the indexed dataset."
    first = results[0]
    intro = f"My best pick is {first['name']} in {first['city']}, {first['country']}."
    if relaxed:
        intro += " I relaxed the softer price or meal constraint because the exact combination returned no restaurants."
    lines = [
        intro,
        f"It has a {first['rating']} rating, is in the {first['price']} price bucket, and matches: {', '.join(first.get('reasons', []))}.",
        "Good alternatives:",
    ]
    for item in results[:3]:
        lines.append(f"{item['rank']}. {item['name']} - rating {item['rating']}, {item['price']}, {item['tags']}.")
    constraints = ", ".join(value for value in [
        f"city={parsed.city}" if parsed.city else "",
        f"country={parsed.country}" if parsed.country else "",
        f"price={parsed.price_bucket}" if parsed.price_bucket else "",
        f"meal={parsed.meal}" if parsed.meal else "",
    ] if value)
    if constraints:
        lines.append(f"Applied constraints: {constraints}.")
    return "\n".join(lines)


def check_groundedness(answer: str, allowed_names: list[str]) -> tuple[bool, list[str]]:
    unsupported: list[str] = []
    allowed_lower = {name.lower() for name in allowed_names}
    for line in answer.splitlines():
        if not line.strip():
            continue
        possible = re.findall(r"(?:^|\d+\.\s)([A-Z][A-Za-z0-9 '&.,-]{2,80})", line)
        for name in possible:
            cleaned = name.strip(" .,-")
            if cleaned and len(cleaned.split()) <= 8 and not any(cleaned.lower() in allowed for allowed in allowed_lower):
                GENERIC_ALLOWED_PHRASES = {
                    "my best pick",
                    "best pick",
                    "good alternatives",
                    "alternatives",
                    "applied constraints",
                    "restaurant",
                    "location",
                    "rating",
                    "price",
                    "cuisine",
                    "tags",
                    "meals",
                    "reasons",
                }
                if cleaned.lower() not in GENERIC_ALLOWED_PHRASES:
                    unsupported.append(cleaned)
    return len(unsupported) == 0, unsupported


def generate_grounded_answer(query: str, results: list[dict[str, object]], parsed: ParsedQuery, relaxed: bool) -> tuple[str, str, bool]:
    log_step("RAG", "build evidence context and generate grounded answer")
    fallback = template_answer(query, results, parsed, relaxed)
    if not results or not load_rag_generator():
        return fallback, "deterministic grounded template", True
    try:
        import torch
        model = _COMPONENTS["rag_model"]
        tokenizer = _COMPONENTS["rag_tokenizer"]
        context = build_rag_context(results)
        messages = [
            {"role": "system", "content": "You are TableWise. Recommend only restaurants from the provided evidence. Do not invent names, ratings, addresses, or features. Keep the answer concise."},
            {"role": "user", "content": f"User query: {query}\n\nEvidence:\n{context}\n\nAnswer with the best pick and two alternatives."},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Use only this evidence.\n{context}\n\nQuery: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        log_step("Parse", "generating structured JSON with fine-tuned SLM")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=RAG_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        grounded, unsupported = check_groundedness(answer, [item["name"] for item in results])
        if not grounded:
            log_step("RAG", f"unsupported mentions detected: {unsupported}; using template fallback")
            return fallback, f"template fallback after unsupported mentions: {unsupported}", True
        log_step("RAG", "generated answer passed groundedness check")
        return answer, "Qwen RAG generator + groundedness check", True
    except Exception as error:
        _COMPONENT_STATUS["rag_generator"] = f"generation failed: {error}; using deterministic template"
        return fallback, "deterministic grounded template after generator failure", True


def component_status() -> dict[str, object]:
    return {
        "slm_parser": _COMPONENT_STATUS.get("slm_parser"),
        "embedding_model": _COMPONENT_STATUS.get("embedding_model"),
        "faiss": _COMPONENT_STATUS.get("faiss"),
        "reward_model": _COMPONENT_STATUS.get("reward_model"),
        "rag_generator": _COMPONENT_STATUS.get("rag_generator"),
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


def search_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    query = clean(params.get("q", [""])[0])
    log_step("HTTP /api/search", f"new request: {query}")
    limit = int(params.get("limit", [str(DEFAULT_LIMIT)])[0] or DEFAULT_LIMIT)
    if not query:
        return {"error": "Please enter a natural-language restaurant query."}
    try:
        return run_full_pipeline(query, limit=limit)
    except Exception as error:
        log_step("PIPELINE ERROR", str(error))
        if os.environ.get("TABLEWISE_ALLOW_LIGHT_FALLBACK", "").strip().lower() not in {"1", "true", "yes"}:
            return {
                "error": str(error),
                "query": query,
                "parsed": {},
                "parser": "unavailable",
                "scanned": 0,
                "filtered": 0,
                "results": [],
                "answer": str(error),
                "relaxed": False,
                "pipeline": component_status(),
            }
        return light_csv_fallback(query, limit)


def light_csv_fallback(query: str, limit: int) -> dict[str, object]:
    parsed = parse_query_rules(query)
    query_terms = re.findall(r"[\w']+", norm(query))
    winners: list[tuple[float, dict[str, str], list[str]]] = []
    scanned = 0
    filtered = 0
    if not DATA_PATH.exists():
        return {"error": f"Dataset not found at {DATA_PATH}"}
    with DATA_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scanned += 1
            if parsed.city and norm(parsed.city) not in norm(row.get("city_filled") or row.get("city")):
                continue
            filtered += 1
            score = safe_float(row.get("rating")) * 1.3 + safe_float(row.get("popularity_score")) * 4
            searchable = norm(" ".join([row.get("profile_text", ""), row.get("top_tags_text", ""), row.get("meals_text", "")]))
            score += sum(1.4 for term in query_terms if len(term) > 2 and term in searchable)
            reasons = ["light CSV fallback result"]
            winners.append((score, row, reasons))
            winners.sort(key=lambda value: value[0], reverse=True)
            winners = winners[:limit]
    results = []
    for rank, (score, row, reasons) in enumerate(winners, start=1):
        results.append({
            "rank": rank,
            "score": round(score, 2),
            "name": clean(row.get("name")) or "Unknown restaurant",
            "city": clean(row.get("city_filled") or row.get("city")),
            "country": clean(row.get("country")),
            "address": clean(row.get("address")),
            "latitude": clean(row.get("latitude")),
            "longitude": clean(row.get("longitude")),
            "rating": clean(row.get("rating")) or "Unknown",
            "price": clean(row.get("price_bucket")) or "Unknown",
            "tags": clean(row.get("top_tags_text") or row.get("cuisines_text")) or "Unknown",
            "meals": clean(row.get("meals_text")) or "Unknown",
            "features": clean(row.get("features_text")) or "Unknown",
            "popularity": clean(row.get("popularity_detailed")) or "Unknown",
            "profile": clean(row.get("short_profile") or row.get("profile_text")),
            "reasons": reasons,
        })
    return {
        "query": query,
        "parsed": parsed.__dict__,
        "parser": "light fallback",
        "scanned": scanned,
        "filtered": filtered,
        "results": results,
        "answer": template_answer(query, results, parsed),
        "relaxed": False,
        "pipeline": component_status(),
    }


def split_tags(value: str) -> set[str]:
    return {part.strip().lower() for part in clean(value).split(",") if part.strip()}


def similar_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    favorite = {
        "name": clean(params.get("name", [""])[0]),
        "city": clean(params.get("city", [""])[0]),
        "country": clean(params.get("country", [""])[0]),
        "address": clean(params.get("address", [""])[0]),
        "latitude": clean(params.get("latitude", [""])[0]),
        "longitude": clean(params.get("longitude", [""])[0]),
        "rating": clean(params.get("rating", [""])[0]),
        "price": normalize_price(clean(params.get("price", [""])[0])),
        "tags": split_tags(params.get("tags", [""])[0]),
        "meals": split_tags(params.get("meals", [""])[0]),
        "features": clean(params.get("features", [""])[0]),
        "popularity": clean(params.get("popularity", [""])[0]),
        "profile": clean(params.get("profile", [""])[0]),
    }
    if not DATA_PATH.exists():
        return {"error": f"Dataset not found at {DATA_PATH}"}
    limit = int(params.get("limit", [str(DEFAULT_LIMIT)])[0] or DEFAULT_LIMIT)
    winners: list[tuple[float, dict[str, str], list[str]]] = []
    scanned = 0
    with DATA_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scanned += 1
            if favorite["name"] and norm(row.get("name")) == norm(favorite["name"]):
                continue
            row_city = norm(row.get("city_filled") or row.get("city"))
            row_country = norm(row.get("country"))
            row_price = normalize_price(row.get("price_bucket") or "")
            row_tags = split_tags(row.get("top_tags_text") or row.get("cuisines_text") or "")
            row_meals = split_tags(row.get("meals_text") or "")
            rating = safe_float(row.get("rating"))
            popularity = safe_float(row.get("popularity_score"))
            score = rating * 1.6 + popularity * 5
            reasons: list[str] = []
            if favorite["city"] and norm(favorite["city"]) == row_city:
                score += 12
                reasons.append(f"also in {row.get('city_filled') or row.get('city')}")
            if favorite["country"] and norm(favorite["country"]) == row_country:
                score += 4
                reasons.append(f"in {row.get('country')}")
            if favorite["price"] and favorite["price"] == row_price:
                score += 7
                reasons.append(f"same {row_price} price range")
            tag_overlap = sorted(favorite["tags"] & row_tags)
            meal_overlap = sorted(favorite["meals"] & row_meals)
            if tag_overlap:
                score += min(len(tag_overlap), 4) * 5
                reasons.append(f"similar cuisine: {', '.join(tag_overlap[:3])}")
            if meal_overlap:
                score += min(len(meal_overlap), 2) * 3
                reasons.append(f"similar meals: {', '.join(meal_overlap[:2])}")
            if rating >= 4.5:
                reasons.append(f"strong rating: {rating:g}/5")
            if score <= 0:
                continue
            winners.append((score, row, reasons[:4] or ["similar restaurant profile"]))
            winners.sort(key=lambda value: value[0], reverse=True)
            winners = winners[:limit]
    results = []
    for rank, (score, row, reasons) in enumerate(winners[:limit], start=1):
        results.append({
            "rank": rank,
            "score": round(score, 2),
            "name": clean(row.get("name")) or "Unknown restaurant",
            "city": clean(row.get("city_filled") or row.get("city")),
            "country": clean(row.get("country")),
            "address": clean(row.get("address")),
            "latitude": clean(row.get("latitude")),
            "longitude": clean(row.get("longitude")),
            "rating": clean(row.get("rating")) or "Unknown",
            "price": clean(row.get("price_bucket")) or "Unknown",
            "tags": clean(row.get("top_tags_text") or row.get("cuisines_text")) or "Unknown",
            "meals": clean(row.get("meals_text")) or "Unknown",
            "features": clean(row.get("features_text")) or "Unknown",
            "popularity": clean(row.get("popularity_detailed")) or "Unknown",
            "profile": clean(row.get("short_profile") or row.get("profile_text")),
            "reasons": reasons,
        })
    answer = f"Nice choice. I reranked restaurants that feel close to {favorite['name'] or 'your favorite'}."
    if results:
        first = results[0]
        answer += f"\nThe closest match is {first['name']} in {first['city']}, with rating {first['rating']} and {first['price']} pricing."
    return {"query": f"More like {favorite['name']}", "parsed": {}, "scanned": scanned, "filtered": len(results), "results": results, "answer": answer, "relaxed": False}


def get_preferences_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    liked_json = clean(params.get("liked", ["[]"])[0])
    try:
        liked = json.loads(liked_json)
    except (json.JSONDecodeError, ValueError):
        liked = []
    if not liked:
        return {"query": "Show restaurants according to my preferences", "parsed": {}, "scanned": 0, "filtered": 0, "results": [], "answer": "You haven't liked any restaurants yet.", "relaxed": False}
    def sort_key(restaurant: dict) -> tuple:
        price_order = {"cheap": 0, "mid": 1, "expensive": 2}
        return ((restaurant.get("tags") or "").lower(), price_order.get((restaurant.get("price") or "").lower(), 3), (restaurant.get("city") or "").lower())
    results = []
    for rank, restaurant in enumerate(sorted(liked, key=sort_key), start=1):
        item = dict(restaurant)
        item.update({"rank": rank, "score": 100.0 - rank * 0.5, "reasons": ["liked by the user", "sorted by cuisine, price, and location"]})
        results.append(item)
    return {"query": "Show restaurants according to my preferences", "parsed": {}, "scanned": len(liked), "filtered": len(results), "results": results, "answer": f"Here are your {len(results)} liked restaurants, sorted by cuisine, price, and location.", "relaxed": False}


def get_more_like_preferences(params: dict[str, list[str]]) -> dict[str, object]:
    liked_json = clean(params.get("liked", ["[]"])[0])
    try:
        liked = json.loads(liked_json)
    except (json.JSONDecodeError, ValueError):
        liked = []
    if not liked:
        return {"query": "Show more restaurants like my preferences", "parsed": {}, "scanned": 0, "filtered": 0, "results": [], "answer": "You haven't liked any restaurants yet.", "relaxed": False}
    liked_cuisines = set()
    liked_prices = set()
    liked_cities = set()
    liked_keys = {(norm(r.get("name")), norm(r.get("city")), norm(r.get("country"))) for r in liked}
    for restaurant in liked:
        for tag in str(restaurant.get("tags", "")).split(","):
            if tag.strip():
                liked_cuisines.add(tag.strip().lower())
        if restaurant.get("price"):
            liked_prices.add(normalize_price(restaurant.get("price")))
        if restaurant.get("city"):
            liked_cities.add(norm(restaurant.get("city")))
    candidates: list[tuple[float, dict[str, str], list[str]]] = []
    scanned = 0
    if not DATA_PATH.exists():
        return {"error": f"Dataset not found at {DATA_PATH}"}
    with DATA_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scanned += 1
            row_key = (norm(row.get("name")), norm(row.get("city_filled") or row.get("city")), norm(row.get("country")))
            if row_key in liked_keys:
                continue
            row_tags = norm(row.get("top_tags_text") or row.get("cuisines_text") or "")
            row_price = normalize_price(row.get("price_bucket") or "")
            row_city = norm(row.get("city_filled") or row.get("city"))
            rating = safe_float(row.get("rating"))
            popularity = safe_float(row.get("popularity_score"))
            score = 0.0
            reasons: list[str] = []
            tag_matches = sum(1 for cuisine in liked_cuisines if cuisine in row_tags)
            if tag_matches:
                score += 12 * tag_matches
                reasons.append(f"cuisine: {tag_matches} match")
            if row_price in liked_prices:
                score += 10
                reasons.append(f"price: {row_price}")
            if row_city in liked_cities:
                score += 15
                reasons.append(f"location: {row_city}")
            if rating >= 4.0:
                score += rating * 2
            if popularity > 0:
                score += popularity * 3
            if score > 0:
                candidates.append((score, row, reasons[:3] or ["quality match"]))
    candidates.sort(key=lambda x: -x[0])
    results = []
    for rank, (score, row, reasons) in enumerate(candidates[:DEFAULT_LIMIT], start=1):
        results.append({
            "rank": rank,
            "score": round(score, 2),
            "name": clean(row.get("name")) or "Unknown restaurant",
            "city": clean(row.get("city_filled") or row.get("city")),
            "country": clean(row.get("country")),
            "address": clean(row.get("address")),
            "latitude": clean(row.get("latitude")),
            "longitude": clean(row.get("longitude")),
            "rating": clean(row.get("rating")) or "Unknown",
            "price": clean(row.get("price_bucket")) or "Unknown",
            "tags": clean(row.get("top_tags_text") or row.get("cuisines_text")) or "Unknown",
            "meals": clean(row.get("meals_text")) or "Unknown",
            "features": clean(row.get("features_text")) or "Unknown",
            "popularity": clean(row.get("popularity_detailed")) or "Unknown",
            "profile": clean(row.get("short_profile") or row.get("profile_text")),
            "reasons": reasons,
        })
    return {"query": "Show more restaurants like my preferences", "parsed": {}, "scanned": scanned, "filtered": len(results), "results": results, "answer": f"Found {len(results)} new restaurants similar to your liked ones.", "relaxed": False}


def save_feedback(params: dict[str, list[str]]) -> dict[str, object]:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = FEEDBACK_PATH.exists()
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": clean(params.get("query", [""])[0]),
        "restaurant": clean(params.get("restaurant", [""])[0]),
        "city": clean(params.get("city", [""])[0]),
        "rating": clean(params.get("rating", [""])[0]),
        "feedback": clean(params.get("feedback", [""])[0]),
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return {"ok": True, "saved_to": str(FEEDBACK_PATH)}


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TableWise</title>
  <style>
    :root { --ink:#172033; --muted:#667085; --line:#d8e0e7; --panel:#fff; --accent:#0b7f73; --soft:#e3f3ef; --warm:#fff2dc; --coral:#bd4b3d; --shadow:0 18px 45px rgba(20,38,50,.10); }
    * { box-sizing: border-box; }
    body { margin:0; color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:linear-gradient(135deg, rgba(11,127,115,.10), rgba(189,75,61,.06) 42%, rgba(255,242,220,.68)), #f8fafb; }
    button,input,select { font:inherit; }
    .shell { min-height:100vh; display:grid; grid-template-columns:300px minmax(0,1fr); }
    aside { background:rgba(255,255,255,.92); border-right:1px solid rgba(216,224,231,.9); padding:24px 20px; position:sticky; top:0; height:100vh; overflow:auto; box-shadow:12px 0 35px rgba(31,45,61,.05); }
    main { padding:24px 28px 34px; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:22px; padding-bottom:18px; border-bottom:1px solid rgba(216,224,231,.78); }
    .mark { width:46px; height:46px; border-radius:12px; display:grid; place-items:center; background:linear-gradient(135deg, var(--accent), #0f9f8c); color:#fff; font-weight:900; box-shadow:0 12px 24px rgba(11,127,115,.28); }
    h1,h2,h3 { margin:0; } h1{font-size:23px;} h2,h3{font-size:18px;} .subtitle{margin:2px 0 0; color:var(--muted); font-size:13px;}
    .panel { background:rgba(255,255,255,.88); border:1px solid rgba(216,224,231,.95); border-radius:14px; box-shadow:0 8px 24px rgba(20,38,50,.05); }
    .side-title { color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.04em; margin:18px 0 10px; }
    .examples { display:flex; flex-wrap:wrap; gap:8px; }
    .example,.secondary,.mini-action { min-height:34px; border-radius:999px; border:1px solid #a9d8d0; background:rgba(227,243,239,.88); color:#064e46; padding:0 12px; font-size:12px; font-weight:800; cursor:pointer; }
    button { border:0; border-radius:11px; background:linear-gradient(135deg, var(--accent), #118f7d); color:white; min-height:44px; padding:0 16px; font-weight:850; cursor:pointer; box-shadow:0 12px 24px rgba(11,127,115,.18); }
    button:disabled { opacity:.7; cursor:wait; }
    .saved-panel { margin-top:20px; padding-top:16px; border-top:1px solid rgba(216,224,231,.78); }
    .saved-head { display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.04em; margin-bottom:10px; }
    .saved-list{display:grid; gap:8px;} .saved-empty{color:var(--muted); font-size:13px; padding:10px 0;}
    .saved-item{border:1px solid rgba(216,224,231,.85); border-radius:12px; background:rgba(255,255,255,.68); padding:10px; display:grid; gap:8px; color:var(--ink); font-size:13px; cursor:pointer; text-align:left;}
    .saved-item span{color:var(--muted); font-size:12px;} .saved-actions{display:flex; gap:6px; flex-wrap:wrap;}
    .hero{display:grid; grid-template-columns:minmax(0,1fr) 190px; gap:14px; align-items:stretch; padding:20px; margin-bottom:16px; background:linear-gradient(135deg, rgba(255,255,255,.95), rgba(255,248,238,.78) 55%, rgba(227,243,239,.72)); box-shadow:var(--shadow);}
    .query-label{font-size:12px; font-weight:850; color:var(--muted); margin-bottom:8px;}
    .query-input{width:100%; min-height:58px; font-size:16px; border:1px solid rgba(11,127,115,.22); border-radius:12px; padding:11px 12px; outline:none; background:rgba(255,255,255,.92);}
    .metrics{display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:12px; margin-bottom:16px;} .metric{padding:16px;} .metric strong{display:block; font-size:24px;} .metric span{color:var(--muted); font-size:12px; font-weight:800;}
    .pipeline{display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); gap:10px; margin-bottom:16px;} .step{padding:13px; border-radius:14px; border:1px solid rgba(216,224,231,.92); background:rgba(255,255,255,.74);} .step strong{display:block; font-size:13px; margin-bottom:4px;} .step span{color:var(--muted); font-size:12px; line-height:1.35;}
    .answer{padding:22px; margin-bottom:16px; background:linear-gradient(135deg, rgba(255,255,255,.96), rgba(255,248,238,.86)); box-shadow:var(--shadow);} .chat-log{display:grid; gap:10px; margin-top:14px;} .chat-empty{color:var(--muted);} .message{max-width:82%; padding:12px 14px; border-radius:14px; line-height:1.5; white-space:pre-line; font-size:14px;} .message.user{justify-self:end; background:var(--accent); color:white;} .message.assistant{justify-self:start; background:rgba(255,255,255,.82); border:1px solid rgba(216,224,231,.9); color:#344054;}
    .results-head{display:flex; justify-content:space-between; gap:12px; align-items:center; margin:8px 0 12px;} .status{color:var(--muted); font-size:13px; text-align:right; max-width:560px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
    .results-tools{display:flex; justify-content:space-between; gap:12px; align-items:center; margin:0 0 14px;} .active-filters{display:flex; flex-wrap:wrap; gap:8px; min-height:32px; align-items:center;} .filter-chip{border:1px solid rgba(11,127,115,.20); background:rgba(227,243,239,.78); color:#07584f; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:850;} .sort-control{width:178px; min-height:38px; padding:8px 10px; font-size:13px; border:1px solid var(--line); border-radius:12px; background:white;}
    .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:18px;} .card{padding:0; overflow:hidden; background:linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.90)); box-shadow:0 16px 35px rgba(16,24,40,.08);} .card-top{display:flex; justify-content:space-between; gap:14px; align-items:flex-start; padding:16px 18px 12px; border-top:4px solid var(--accent);} .rank-badge{min-width:48px; height:48px; border-radius:14px; display:grid; place-items:center; background:linear-gradient(135deg,var(--soft),#fff7e8); color:#07584f; font-size:18px; font-weight:950;} .score{color:var(--muted); font-size:12px; font-weight:800; margin-top:4px; text-align:right;} .place{margin-top:5px; color:var(--muted); font-size:13px;} .card-body{padding:0 18px 16px; display:grid; gap:12px;} .chips{display:flex; gap:7px; flex-wrap:wrap;} .chip{border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:#fbfcfd; color:#344054; font-size:12px; font-weight:750;} .rating{border-color:#f3cc7a; background:#fff8e6; color:#a05a00;} .tags{color:#344054; line-height:1.35;} .why{padding:12px; border:1px solid #e4e7ec; border-radius:12px; background:rgba(248,250,251,.78); color:#344054; font-size:13px; line-height:1.45;} .why-title{color:var(--muted); font-size:11px; font-weight:900; text-transform:uppercase; margin-bottom:7px;} .why-list{display:grid; gap:5px;} .address,.popularity{color:var(--muted); font-size:13px;} .card-actions{display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;} a.mini-action{text-decoration:none; display:inline-flex; align-items:center;} .empty{padding:30px; text-align:center; color:var(--muted);}
    .toast{position:fixed; right:22px; bottom:22px; z-index:20; min-width:260px; max-width:360px; padding:14px 16px; border-radius:14px; background:#102a27; color:white; box-shadow:0 18px 42px rgba(16,42,39,.28); font-weight:750; transform:translateY(18px); opacity:0; pointer-events:none; transition:opacity .18s ease, transform .18s ease;} .toast.show{opacity:1; transform:translateY(0);}
    @media(max-width:920px){.shell{grid-template-columns:1fr;} aside{position:static; height:auto;} main{padding:16px;} .hero{grid-template-columns:1fr;} .metrics{grid-template-columns:repeat(2,1fr);} .pipeline{grid-template-columns:1fr;} .status{text-align:left; white-space:normal;}}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand"><div class="mark">TW</div><div><h1>TableWise</h1><p class="subtitle">Full pipeline demo</p></div></div>
      <div class="side-title">Quick searches</div>
      <div class="examples">
        <button class="example" data-q="cheap vegetarian italian restaurant in rome">Italian Rome</button>
        <button class="example" data-q="vegetarian brunch in barcelona">Veg brunch</button>
        <button class="example" data-q="seafood dinner in lisbon">Seafood</button>
        <button class="example" data-q="fine dining in milan">Fine dining</button>
      </div>
      <section class="saved-panel">
        <div class="saved-head"><span>Your preferences</span><button id="clearSavedButton" class="secondary" type="button">Clear</button></div>
        <div class="saved-head"><span>Liked (<span id="likedCount">0</span>)</span></div><div class="saved-list" id="likedList"><div class="saved-empty">No liked restaurants yet.</div></div>
        <div class="saved-head" style="margin-top:18px"><span>Saved (<span id="savedCountSide">0</span>)</span></div><div class="saved-list" id="savedList"><div class="saved-empty">No saved restaurants yet.</div></div>
        <div class="saved-head" style="margin-top:18px"><span>Disliked (<span id="dislikedCount">0</span>)</span></div><div class="saved-list" id="dislikedList"><div class="saved-empty">No disliked restaurants yet.</div></div>
      </section>
    </aside>
    <main>
      <form class="panel hero" id="searchForm"><div><div class="query-label">Ask TableWise</div><input class="query-input" id="q" placeholder="cheap vegetarian Italian restaurant in Rome" autocomplete="off" /></div><button class="search-button" id="searchButton" type="submit">Search</button></form>
      <section class="metrics"><div class="panel metric"><strong id="topPicks">0</strong><span>Top picks</span></div><div class="panel metric"><strong id="bestRating">-</strong><span>Best rating</span></div><div class="panel metric"><strong id="priceVibe">Any</strong><span>Price vibe</span></div><div class="panel metric"><strong id="savedCount">0</strong><span>Saved picks</span></div></section>
      <section class="pipeline"><div class="step"><strong>1. SLM Parse</strong><span id="pipeParse">Waiting</span></div><div class="step"><strong>2. FAISS Retrieve</strong><span id="pipeRetrieve">Indexed profiles</span></div><div class="step"><strong>3. Rerank</strong><span id="pipeRerank">Semantic + metadata</span></div><div class="step"><strong>4. Reward</strong><span id="pipeReward">Feedback model</span></div><div class="step"><strong>5. RAG</strong><span id="pipeRag">Grounded answer</span></div></section>
      <section class="panel answer"><h2>TableWise chat</h2><div id="chatLog" class="chat-log"><div class="chat-empty">Ask a natural-language restaurant question.</div></div></section>
      <div class="results-head"><h2>Ranked restaurants</h2><div class="status" id="status">Ready</div></div>
      <div class="results-tools"><div class="active-filters" id="activeFilters"></div><select class="sort-control" id="sortSelect"><option value="relevance">Sort by relevance</option><option value="rating">Sort by rating</option><option value="score">Sort by score</option><option value="name">Sort by name</option></select></div>
      <section class="grid" id="results"><div class="panel empty">No restaurants loaded yet.</div></section>
    </main>
  </div>
  <div class="toast" id="toast"></div>
<script>
const $ = (id) => document.getElementById(id);
let currentResults = [];
let savedPicks = JSON.parse(localStorage.getItem("tablewiseSaved") || "[]");
let likedRestaurants = JSON.parse(localStorage.getItem("tablewiseLiked") || "[]");
let dislikedRestaurants = JSON.parse(localStorage.getItem("tablewiseDisliked") || "[]");
let chatMessages = [];
function escapeHtml(value){return String(value || "").replace(/[&<>"']/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));}
function ratingValue(r){const v=Number.parseFloat(r.rating); return Number.isFinite(v)?v:0;}
function samePlace(a,b){return a&&b&&a.name===b.name&&a.city===b.city&&a.country===b.country;}
function isSaved(r){return savedPicks.some((x)=>samePlace(x,r));} function isLiked(r){return likedRestaurants.some((x)=>samePlace(x,r));} function isDisliked(r){return dislikedRestaurants.some((x)=>samePlace(x,r));}
function saveState(){localStorage.setItem("tablewiseSaved",JSON.stringify(savedPicks)); localStorage.setItem("tablewiseLiked",JSON.stringify(likedRestaurants)); localStorage.setItem("tablewiseDisliked",JSON.stringify(dislikedRestaurants));}
function showToast(t){$("toast").textContent=t; $("toast").classList.add("show"); clearTimeout(showToast.timer); showToast.timer=setTimeout(()=>$("toast").classList.remove("show"),2200);}
function mapUrl(r){return r.latitude&&r.longitude?`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${r.latitude},${r.longitude}`)}`:`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${r.name} ${r.address || r.city}`)}`;}
function renderSaved(){
  $("savedCount").textContent=savedPicks.length; $("savedCountSide").textContent=savedPicks.length; $("likedCount").textContent=likedRestaurants.length; $("dislikedCount").textContent=dislikedRestaurants.length;
  function listHtml(arr, kind){return arr.length?arr.map((x,i)=>`<button class="saved-item" type="button" data-kind="${kind}" data-index="${i}"><strong>${escapeHtml(x.name)}</strong><span>${escapeHtml(x.city)}, ${escapeHtml(x.country)} - rating ${escapeHtml(x.rating)}</span><div class="saved-actions"><span class="mini-action remove" data-kind="${kind}" data-index="${i}">Remove</span></div></button>`).join(""):`<div class="saved-empty">No ${kind} restaurants yet.</div>`;}
  $("likedList").innerHTML=listHtml(likedRestaurants,"liked"); $("savedList").innerHTML=listHtml(savedPicks,"saved"); $("dislikedList").innerHTML=listHtml(dislikedRestaurants,"disliked");
  document.querySelectorAll(".remove").forEach((b)=>b.addEventListener("click",(e)=>{e.stopPropagation(); const i=Number(b.dataset.index); const k=b.dataset.kind; if(k==="liked")likedRestaurants.splice(i,1); if(k==="saved")savedPicks.splice(i,1); if(k==="disliked")dislikedRestaurants.splice(i,1); saveState(); renderSaved(); renderResults();}));
}
function toggleLike(r){ if(isLiked(r)) likedRestaurants=likedRestaurants.filter((x)=>!samePlace(x,r)); else { dislikedRestaurants=dislikedRestaurants.filter((x)=>!samePlace(x,r)); likedRestaurants=[r,...likedRestaurants]; } saveState(); renderSaved(); renderResults(); sendFeedback(r,"good"); }
function toggleDislike(r){ if(isDisliked(r)) dislikedRestaurants=dislikedRestaurants.filter((x)=>!samePlace(x,r)); else { likedRestaurants=likedRestaurants.filter((x)=>!samePlace(x,r)); dislikedRestaurants=[r,...dislikedRestaurants]; } saveState(); renderSaved(); renderResults(); sendFeedback(r,"bad"); }
function toggleSave(r){ if(isSaved(r)) savedPicks=savedPicks.filter((x)=>!samePlace(x,r)); else savedPicks=[r,...savedPicks].slice(0,8); saveState(); renderSaved(); renderResults(); }
function renderChat(){ $("chatLog").innerHTML=chatMessages.length?chatMessages.map((m)=>`<div class="message ${m.role}">${escapeHtml(m.text)}</div>`).join(""):`<div class="chat-empty">Ask a natural-language restaurant question.</div>`; }
function pushChat(role,text){ if(!text)return; chatMessages.push({role,text}); chatMessages=chatMessages.slice(-6); renderChat(); }
function updateStats(data){const res=data.results||[]; const best=res.reduce((m,x)=>Math.max(m,ratingValue(x)),0); const prices=res.map((x)=>x.price).filter(Boolean); $("topPicks").textContent=res.length; $("bestRating").textContent=best?best.toFixed(1):"-"; $("priceVibe").textContent=prices[0]||"Any"; $("savedCount").textContent=savedPicks.length;}
function updatePipeline(data){const p=data.parsed||{}; const pipe=data.pipeline||{}; $("pipeParse").textContent=`${data.parser || "parser"}: ${[p.city,p.country,p.price_bucket,p.meal].filter(Boolean).join(" / ") || "open query"}`; $("pipeRetrieve").textContent=pipe.faiss || `${data.results.length} results`; $("pipeRerank").textContent="Hybrid reranking applied"; $("pipeReward").textContent=pipe.reward_model || "reward model not loaded"; $("pipeRag").textContent=data.answer_method || pipe.rag_generator || "grounded answer";}
function renderActiveFilters(p){const chips=[]; if(p.city)chips.push(`City: ${p.city}`); if(p.country)chips.push(`Country: ${p.country}`); if(p.price_bucket)chips.push(`Price: ${p.price_bucket}`); if(p.meal)chips.push(`Meal: ${p.meal}`); if(p.tags&&p.tags.length)chips.push(`Tags: ${p.tags.join(", ")}`); $("activeFilters").innerHTML=chips.length?chips.map((c)=>`<span class="filter-chip">${escapeHtml(c)}</span>`).join(""):`<span class="filter-chip">Full pipeline search</span>`;}
function sortedResults(){const mode=$("sortSelect").value; return currentResults.filter((x)=>!isDisliked(x)).sort((a,b)=>{if(mode==="rating")return ratingValue(b)-ratingValue(a)||b.score-a.score; if(mode==="score"||mode==="relevance")return b.score-a.score; if(mode==="name")return a.name.localeCompare(b.name); return a.rank-b.rank;});}
function card(r,i){const reasons=(r.reasons||[]).map((x)=>`<div>${escapeHtml(x)}</div>`).join(""); return `<article class="panel card"><div class="card-top"><div><h3>${escapeHtml(r.name)}</h3><div class="place">${escapeHtml(r.city)}, ${escapeHtml(r.country)}</div></div><div><div class="rank-badge">#${r.rank}</div><div class="score">${r.score}</div></div></div><div class="card-body"><div class="chips"><span class="chip rating">Rating ${escapeHtml(r.rating)}</span><span class="chip">${escapeHtml(r.price)}</span><span class="chip">Reward ${escapeHtml(r.reward_score ?? "-")}</span></div><div class="tags">${escapeHtml(r.tags)}</div><div class="why"><div class="why-title">Why this matches</div><div class="why-list">${reasons}</div></div><div class="address">${escapeHtml(r.address || r.popularity)}</div><div class="card-actions"><button class="mini-action save-action" data-index="${i}">${isSaved(r)?"Saved":"Save"}</button><button class="mini-action like-action" data-index="${i}">${isLiked(r)?"Liked":"Good"}</button><button class="mini-action dislike-action" data-index="${i}">${isDisliked(r)?"Disliked":"Not right"}</button><button class="mini-action similar-action" data-index="${i}">More like this</button><a class="mini-action" href="${mapUrl(r)}" target="_blank" rel="noreferrer">Map</a></div></div></article>`;}
function renderResults(){const res=sortedResults(); $("results").innerHTML=res.length?res.map((r,i)=>card(r,i)).join(""):`<div class="panel empty">No matching restaurants.</div>`; document.querySelectorAll(".save-action").forEach((b)=>b.addEventListener("click",()=>toggleSave(sortedResults()[Number(b.dataset.index)]))); document.querySelectorAll(".like-action").forEach((b)=>b.addEventListener("click",()=>toggleLike(sortedResults()[Number(b.dataset.index)]))); document.querySelectorAll(".dislike-action").forEach((b)=>b.addEventListener("click",()=>toggleDislike(sortedResults()[Number(b.dataset.index)]))); document.querySelectorAll(".similar-action").forEach((b)=>b.addEventListener("click",()=>rerankLike(sortedResults()[Number(b.dataset.index)])));}
function setBusy(v){$("searchButton").disabled=v; $("searchButton").textContent=v?"Searching...":"Search"; if(v)$("status").textContent="Running full pipeline...";}
async function search(){const q=$("q").value.trim(); if(!q)return; setBusy(true); const params=new URLSearchParams({q,limit:"9"}); try{const r=await fetch(`/api/search?${params.toString()}`); const data=await r.json(); if(data.error)throw new Error(data.error); currentResults=data.results; updateStats(data); updatePipeline(data); renderActiveFilters(data.parsed||{}); renderResults(); pushChat("user",q); pushChat("assistant",data.answer); $("status").textContent=`${data.filtered} candidates after filtering; grounded=${data.grounded ?? true}`;}catch(e){$("status").textContent=e.message; $("results").innerHTML=`<div class="panel empty">${escapeHtml(e.message)}</div>`; pushChat("assistant",e.message);}finally{setBusy(false);}}
async function rerankLike(r){setBusy(true); pushChat("user",`Show me more restaurants like ${r.name}.`); const params=new URLSearchParams({name:r.name,city:r.city,country:r.country,address:r.address,latitude:r.latitude,longitude:r.longitude,rating:r.rating,price:r.price,tags:r.tags,meals:r.meals,features:r.features,popularity:r.popularity,profile:r.profile,limit:"9"}); try{const resp=await fetch(`/api/similar?${params.toString()}`); const data=await resp.json(); currentResults=data.results; updateStats(data); renderResults(); pushChat("assistant",data.answer); $("status").textContent=`Reranked from ${r.name}`;}catch(e){pushChat("assistant",e.message);}finally{setBusy(false);}}
async function sendFeedback(r,feedback){showToast(feedback==="good"?`You liked ${r.name}`:`Feedback saved for ${r.name}`); const params=new URLSearchParams({query:$("q").value.trim(),restaurant:r.name,city:r.city,rating:r.rating,feedback}); try{await fetch(`/api/feedback?${params.toString()}`);}catch(e){}}
$("searchForm").addEventListener("submit",(e)=>{e.preventDefault(); search();}); $("sortSelect").addEventListener("change",renderResults); $("clearSavedButton").addEventListener("click",()=>{savedPicks=[]; likedRestaurants=[]; dislikedRestaurants=[]; saveState(); renderSaved(); renderResults();}); document.querySelectorAll(".example").forEach((b)=>b.addEventListener("click",()=>{$("q").value=b.dataset.q; search();})); renderSaved(); renderChat();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed_url.path == "/api/search":
            self.send_json(search_restaurants(parse_qs(parsed_url.query)))
            return
        if parsed_url.path == "/api/similar":
            self.send_json(similar_restaurants(parse_qs(parsed_url.query)))
            return
        if parsed_url.path == "/api/preferences":
            self.send_json(get_preferences_restaurants(parse_qs(parsed_url.query)))
            return
        if parsed_url.path == "/api/more-like-preferences":
            self.send_json(get_more_like_preferences(parse_qs(parsed_url.query)))
            return
        if parsed_url.path == "/api/feedback":
            self.send_json(save_feedback(parse_qs(parsed_url.query)))
            return
        if parsed_url.path == "/api/status":
            self.send_json(component_status())
            return
        self.send_json({"error": "Not found"}, status=404)


def main() -> None:
    log_step("Startup", f"ROOT={ROOT}")
    log_step("Startup", f"ARTIFACTS_DIR={ARTIFACTS_DIR}")
    log_step("Startup", f"FAISS_INDEX_PATH={FAISS_INDEX_PATH}")
    log_step("Startup", f"SLM_ADAPTER_PATH={SLM_ADAPTER_PATH}")
    log_step("Startup", f"REWARD_MODEL_PATH={REWARD_MODEL_PATH}")
    log_step("Startup", "Use TABLEWISE_VERBOSE=0 to hide debug logs")
    port = 8501
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"TableWise full-pipeline UI running at http://127.0.0.1:{port}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Dataset: {DATA_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
