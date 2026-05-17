from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "app_restaurants_sample.csv"
FULL_DATA_PATH = ROOT / "data" / "data_new" / "processed" / "restaurants_processed.csv"
DATA_PATH = SAMPLE_PATH if SAMPLE_PATH.exists() else FULL_DATA_PATH
FEEDBACK_PATH = ROOT / "data" / "artifacts" / "feedback" / "ui_feedback.csv"
DEFAULT_LIMIT = 8

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
    "italian",
    "pizza",
    "french",
    "spanish",
    "greek",
    "portuguese",
    "seafood",
    "vegetarian",
    "vegan",
    "gluten free",
    "sushi",
    "asian",
    "indian",
    "mexican",
    "mediterranean",
    "cafe",
    "coffee",
    "romantic",
    "family",
    "fine dining",
    "cheap eats",
    "healthy",
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


def clean(value: str | None) -> str:
    return (value or "").strip()


def norm(value: str | None) -> str:
    return clean(value).lower()


def contains_any(text: str, needles: set[str]) -> bool:
    return any(needle in text for needle in needles)


def extract_excluded_tags(query: str) -> tuple[str, ...]:
    q = norm(query)
    excluded: list[str] = []
    for tag in COMMON_TAGS:
        if re.search(rf"\b(?:don't want|dont want|do not want|no|not|without|exclude|excluding)\s+(?:any\s+)?{re.escape(tag)}\b", q):
            excluded.append(tag)
    return tuple(sorted(set(excluded)))


def parse_query(query: str) -> ParsedQuery:
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


def normalize_price(value: str) -> str:
    value = norm(value)
    if value in {"mid-range", "moderate", "medium"}:
        return "mid"
    return value


def safe_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan"):
            return default
        return float(value)
    except ValueError:
        return default


def score_row(row: dict[str, str], query_terms: list[str], parsed: ParsedQuery) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    city = norm(row.get("city_filled") or row.get("city"))
    country = norm(row.get("country"))
    price = normalize_price(row.get("price_bucket") or "")
    meals = norm(row.get("meals_text"))
    tags = norm(row.get("top_tags_text") or row.get("cuisines_text"))
    diets = norm(row.get("special_diets_text"))
    features = norm(row.get("features_text"))
    profile = norm(row.get("profile_text"))

    if parsed.city and norm(parsed.city) in city:
        score += 18
        reasons.append(f"located in {row.get('city_filled') or row.get('city')}")
    elif parsed.city:
        score -= 8

    if parsed.country and norm(parsed.country) in country:
        score += 8
        reasons.append(f"in {row.get('country')}")

    if parsed.price_bucket and parsed.price_bucket == price:
        score += 9
        reasons.append(f"{parsed.price_bucket} price range")
    elif parsed.price_bucket:
        score -= 3

    if parsed.meal and parsed.meal in meals:
        score += 6
        reasons.append(f"serves {parsed.meal}")

    for tag in parsed.tags:
        if tag in tags or tag in diets or tag in features:
            score += 5
            reasons.append(f"matches {tag}")

    matched_terms = 0
    searchable = f"{profile} {tags} {meals} {diets} {features}"
    for term in query_terms:
        if len(term) > 2 and term in searchable:
            matched_terms += 1
    score += min(matched_terms, 8) * 1.4

    rating = safe_float(row.get("rating"))
    popularity = safe_float(row.get("popularity_score"))
    reviews_total = safe_float(row.get("popularity_total_num"))
    quality = safe_float(row.get("profile_quality_score"))

    score += rating * 1.3
    score += popularity * 4
    score += min(math.log1p(max(reviews_total, 0)), 8) * 0.25
    score += min(quality, 12) * 0.15

    if rating >= 4.5:
        reasons.append(f"strong rating: {rating:g}/5")
    if row.get("awards_text") and norm(row.get("awards_text")) != "unknown":
        score += 2
        reasons.append("award signal")

    if not reasons:
        reasons.append("has overlapping metadata with the query")
    return score, reasons[:4]


def passes_filters(row: dict[str, str], parsed: ParsedQuery, min_rating: float) -> bool:
    if parsed.city and norm(parsed.city) not in norm(row.get("city_filled") or row.get("city")):
        return False
    if parsed.country and norm(parsed.country) not in norm(row.get("country")):
        return False
    if parsed.price_bucket and parsed.price_bucket != normalize_price(row.get("price_bucket") or ""):
        return False
    if parsed.meal and parsed.meal not in norm(row.get("meals_text")):
        return False
    if parsed.exclude_tags:
        searchable = " ".join(
            [
                norm(row.get("top_tags_text") or row.get("cuisines_text") or ""),
                norm(row.get("special_diets_text")),
                norm(row.get("features_text")),
                norm(row.get("profile_text")),
            ]
        )
        if any(excluded in searchable for excluded in parsed.exclude_tags):
            return False
    if min_rating and safe_float(row.get("rating")) < min_rating:
        return False
    return True


def search_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    query = clean(params.get("q", [""])[0])
    parsed = parse_query(query)

    city = clean(params.get("city", [""])[0]) or parsed.city
    country = clean(params.get("country", [""])[0]) or parsed.country
    price = normalize_price(clean(params.get("price", [""])[0]) or parsed.price_bucket)
    meal = clean(params.get("meal", [""])[0]) or parsed.meal
    min_rating = safe_float(params.get("rating", ["0"])[0]) or parsed.min_rating
    limit = int(params.get("limit", [str(DEFAULT_LIMIT)])[0] or DEFAULT_LIMIT)
    parsed = ParsedQuery(
        city=city,
        country=country,
        price_bucket=price,
        meal=meal,
        tags=parsed.tags,
        exclude_tags=parsed.exclude_tags,
        min_rating=min_rating,
    )

    query_terms = re.findall(r"[\w']+", norm(query))
    winners: list[tuple[float, dict[str, str], list[str]]] = []
    scanned = 0
    filtered = 0
    relaxed = False

    if not DATA_PATH.exists():
        return {"error": f"Dataset not found at {DATA_PATH}"}

    def collect(active_parsed: ParsedQuery, active_min_rating: float) -> tuple[list[tuple[float, dict[str, str], list[str]]], int, int]:
        local_winners: list[tuple[float, dict[str, str], list[str]]] = []
        local_scanned = 0
        local_filtered = 0
        with DATA_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                local_scanned += 1
                if not passes_filters(row, active_parsed, active_min_rating):
                    continue
                local_filtered += 1
                score, reasons = score_row(row, query_terms, parsed)
                if query and score <= 0:
                    continue
                local_winners.append((score, row, reasons))
                local_winners.sort(key=lambda value: value[0], reverse=True)
                if len(local_winners) > limit * 4:
                    local_winners = local_winners[:limit]
        return local_winners[:limit], local_scanned, local_filtered

    winners, scanned, filtered = collect(parsed, min_rating)

    if not winners and (parsed.price_bucket or parsed.meal):
        relaxed = True
        relaxed_parsed = ParsedQuery(
            city=parsed.city,
            country=parsed.country,
            tags=parsed.tags,
            exclude_tags=parsed.exclude_tags,
        )
        winners, scanned, filtered = collect(relaxed_parsed, min_rating)

    results = []
    for rank, (score, row, reasons) in enumerate(winners[:limit], start=1):
        results.append(
            {
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
            }
        )

    answer = build_answer(query, results, parsed, relaxed)
    return {
        "query": query,
        "parsed": parsed.__dict__,
        "scanned": scanned,
        "filtered": filtered,
        "results": results,
        "answer": answer,
        "relaxed": relaxed,
    }


def split_tags(value: str) -> set[str]:
    return {part.strip().lower() for part in clean(value).split(",") if part.strip()}


def similar_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    favorite = {
        "name": clean(params.get("name", [""])[0]),
        "city": clean(params.get("city", [""])[0]),
        "country": clean(params.get("country", [""])[0]),
        "price": normalize_price(clean(params.get("price", [""])[0])),
        "tags": split_tags(params.get("tags", [""])[0]),
        "meals": split_tags(params.get("meals", [""])[0]),
    }
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
            if len(winners) > limit * 4:
                winners = winners[:limit]

    results = []
    for rank, (score, row, reasons) in enumerate(winners[:limit], start=1):
        results.append(
            {
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
            }
        )

    answer = f"Nice choice. I reranked restaurants that feel close to {favorite['name'] or 'your favorite'}."
    if results:
        first = results[0]
        answer += f"\nThe closest match is {first['name']} in {first['city']}, with rating {first['rating']} and {first['price']} pricing."
    return {
        "query": f"More like {favorite['name']}",
        "parsed": {"city": favorite["city"], "country": favorite["country"], "price_bucket": favorite["price"], "meal": "", "tags": sorted(favorite["tags"])},
        "scanned": scanned,
        "filtered": len(results),
        "results": results,
        "answer": answer,
        "relaxed": False,
    }


def get_preferences_restaurants(params: dict[str, list[str]]) -> dict[str, object]:
    """Return liked restaurants sorted by: cuisine, price, then location"""
    liked_restaurants_json = clean(params.get("liked", ["[]"])[0])
    try:
        liked = json.loads(liked_restaurants_json)
    except (json.JSONDecodeError, ValueError):
        liked = []
    
    if not liked:
        return {
            "query": "Show restaurants according to my preferences",
            "parsed": {"city": "", "country": "", "price_bucket": "", "meal": "", "tags": []},
            "scanned": 0,
            "filtered": 0,
            "results": [],
            "answer": "You haven't liked any restaurants yet. Try searching for some and mark your favorites as 'Good' to build your preference list.",
            "relaxed": False,
        }
    
    # Sort by: cuisine (tags), price, then location (city)
    def sort_key(restaurant: dict) -> tuple:
        cuisine = (restaurant.get("tags") or "").lower()
        price_order = {"cheap": 0, "mid": 1, "expensive": 2}
        price = price_order.get((restaurant.get("price") or "").lower(), 3)
        city = (restaurant.get("city") or "").lower()
        return (cuisine, price, city)
    
    sorted_restaurants = sorted(liked, key=sort_key)
    
    results = []
    for rank, restaurant in enumerate(sorted_restaurants, start=1):
        results.append({
            "rank": rank,
            "score": 100.0 - (rank * 0.5),  # Highest score for most preferred
            "name": restaurant.get("name", "Unknown"),
            "city": restaurant.get("city", "Unknown"),
            "country": restaurant.get("country", "Unknown"),
            "address": restaurant.get("address", ""),
            "latitude": restaurant.get("latitude", ""),
            "longitude": restaurant.get("longitude", ""),
            "rating": restaurant.get("rating", "Unknown"),
            "price": restaurant.get("price", "Unknown"),
            "tags": restaurant.get("tags", "Unknown"),
            "meals": restaurant.get("meals", "Unknown"),
            "features": restaurant.get("features", "Unknown"),
            "popularity": restaurant.get("popularity", "Unknown"),
            "profile": restaurant.get("profile", ""),
            "reasons": [
                f"cuisine: {restaurant.get('tags', 'N/A')}",
                f"price: {restaurant.get('price', 'N/A')}",
                f"location: {restaurant.get('city', 'N/A')}"
            ],
        })
    
    answer = f"Here are your {len(results)} liked restaurants, sorted by cuisine, price, and location:\n"
    if results:
        for i, result in enumerate(results[:3], 1):
            answer += f"\n{i}. {result['name']} in {result['city']} - {result['price']} price, {result['tags']}"
    
    return {
        "query": "Show restaurants according to my preferences",
        "parsed": {"city": "", "country": "", "price_bucket": "", "meal": "", "tags": []},
        "scanned": len(liked),
        "filtered": len(results),
        "results": results,
        "answer": answer,
        "relaxed": False,
    }


def get_more_like_preferences(params: dict[str, list[str]]) -> dict[str, object]:
    """Find new restaurants similar to liked ones, sorted by: cuisine, price, then location"""
    liked_restaurants_json = clean(params.get("liked", ["[]"])[0])
    try:
        liked = json.loads(liked_restaurants_json)
    except (json.JSONDecodeError, ValueError):
        liked = []
    
    if not liked:
        return {
            "query": "Show more restaurants like my preferences",
            "parsed": {"city": "", "country": "", "price_bucket": "", "meal": "", "tags": []},
            "scanned": 0,
            "filtered": 0,
            "results": [],
            "answer": "You haven't liked any restaurants yet. Try searching for some and mark your favorites as 'Good' to build your preference list.",
            "relaxed": False,
        }
    
    # Extract characteristics from liked restaurants
    liked_cuisines = set()
    liked_prices = set()
    liked_cities = set()
    # Create a set of (name, city, country) tuples to exclude from results
    liked_keys = {(norm(r.get("name")), norm(r.get("city")), norm(r.get("country"))) for r in liked}
    
    for restaurant in liked:
        tags = restaurant.get("tags", "")
        if tags:
            # Extract individual cuisines from comma-separated tags
            for tag in str(tags).split(","):
                cuisine = tag.strip().lower()
                if cuisine:
                    liked_cuisines.add(cuisine)
        price = restaurant.get("price", "")
        if price:
            liked_prices.add(normalize_price(price))
        city = restaurant.get("city", "")
        if city:
            liked_cities.add(norm(city))
    
    # Search database for similar restaurants
    candidates: list[tuple[float, dict[str, str], list[str]]] = []
    scanned = 0
    
    if not DATA_PATH.exists():
        return {"error": f"Dataset not found at {DATA_PATH}"}
    
    with DATA_PATH.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scanned += 1
            
            # Skip already liked restaurants - check by name, city, country combination
            row_name = norm(row.get("name"))
            row_city = norm(row.get("city_filled") or row.get("city"))
            row_country = norm(row.get("country"))
            
            if (row_name, row_city, row_country) in liked_keys:
                continue
            
            score = 0.0
            reasons: list[str] = []
            
            row_tags = norm(row.get("top_tags_text") or row.get("cuisines_text") or "")
            row_price = normalize_price(row.get("price_bucket") or "")
            rating = safe_float(row.get("rating"))
            popularity = safe_float(row.get("popularity_score"))
            
            # Score based on matching characteristics
            # Cuisine matching (strong signal)
            tag_matches = 0
            for cuisine in liked_cuisines:
                if cuisine in row_tags:
                    tag_matches += 1
                    score += 12
            if tag_matches > 0:
                reasons.append(f"cuisine: {tag_matches} match")
            
            # Price matching
            if row_price in liked_prices:
                score += 10
                reasons.append(f"price: {row_price}")
            
            # City/Location matching
            if row_city in liked_cities:
                score += 15
                reasons.append(f"location: {row_city}")
            
            # Quality signals
            if rating >= 4.0:
                score += rating * 2
            
            if popularity > 0:
                score += popularity * 3
            
            # If no specific matches, still consider restaurants with good quality
            if len(reasons) == 0 and (rating >= 3.5 or popularity > 0):
                score = (rating * 1.5) + (popularity * 2)
                reasons.append("quality match")
            
            if score > 0:
                candidates.append((score, row, reasons[:3]))
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: -x[0])
    
    results = []
    for rank, (score, row, reasons) in enumerate(candidates[:9], start=1):
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
    
    answer = f"Found {len(results)} new restaurants similar to your liked ones:\n"
    if results:
        for i, result in enumerate(results[:3], 1):
            answer += f"\n{i}. {result['name']} in {result['city']} - {result['price']} price, {result['tags']}"
            if result['reasons']:
                answer += f"\n   Why: {', '.join(result['reasons'])}"
    else:
        answer = "No new restaurants found matching your preferences. Try liking more restaurants to improve recommendations."
    
    return {
        "query": "Show more restaurants like my preferences",
        "parsed": {"city": "", "country": "", "price_bucket": "", "meal": "", "tags": sorted(list(liked_cuisines))},
        "scanned": scanned,
        "filtered": len(results),
        "results": results,
        "answer": answer,
        "relaxed": False,
    }


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


def build_answer(query: str, results: list[dict[str, object]], parsed: ParsedQuery, relaxed: bool = False) -> str:
    if not results:
        return "I could not find a confident match for those constraints. Try a broader city search or remove one filter."

    first = results[0]
    intro = f"My best pick is {first['name']} in {first['city']}, {first['country']}."
    if relaxed:
        intro += " I relaxed the softer filters because the exact combination returned no restaurants."
    lines = [
        intro,
        f"It has a {first['rating']} rating, sits in the {first['price']} price bucket, and matches: {', '.join(first['reasons'])}.",
    ]
    if len(results) > 1:
        lines.append("Good alternatives:")
    for result in results[:3]:
        lines.append(f"{result['rank']}. {result['name']} - rating {result['rating']}, {result['price']}, {result['tags']}.")
    if parsed.city or parsed.country or parsed.price_bucket or parsed.meal:
        constraints = ", ".join(
            value
            for value in [
                f"city={parsed.city}" if parsed.city else "",
                f"country={parsed.country}" if parsed.country else "",
                f"price={parsed.price_bucket}" if parsed.price_bucket else "",
                f"meal={parsed.meal}" if parsed.meal else "",
            ]
            if value
        )
        lines.append(f"Applied constraints: {constraints}.")
    return "\n".join(lines)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TableWise</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --line: #d8e0e7;
      --paper: #eef3f5;
      --panel: #ffffff;
      --accent: #0b7f73;
      --accent-dark: #07584f;
      --soft: #e3f3ef;
      --warm: #fff2dc;
      --coral: #bd4b3d;
      --warn: #a05a00;
      --shadow: 0 18px 45px rgba(20, 38, 50, .10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(135deg, rgba(11, 127, 115, .10), rgba(189, 75, 61, .06) 42%, rgba(255, 242, 220, .68)),
        linear-gradient(#f8fafb 0 0);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select { font: inherit; }
    .shell { min-height: 100vh; display: grid; grid-template-columns: 340px minmax(0, 1fr); }
    aside {
      background:
        linear-gradient(180deg, rgba(255,255,255,.94), rgba(248, 250, 251, .88)),
        var(--panel);
      border-right: 1px solid rgba(216, 224, 231, .9);
      padding: 24px 20px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      box-shadow: 12px 0 35px rgba(31, 45, 61, .05);
    }
    main { padding: 24px 28px 34px; }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 22px;
      padding: 0 2px 18px;
      border-bottom: 1px solid rgba(216, 224, 231, .78);
    }
    .mark {
      width: 46px; height: 46px; border-radius: 12px;
      display: grid; place-items: center;
      background: linear-gradient(135deg, var(--accent), #0f9f8c);
      color: white;
      font-weight: 900;
      letter-spacing: 0;
      box-shadow: 0 12px 24px rgba(11, 127, 115, .28);
    }
    h1 { margin: 0; font-size: 23px; letter-spacing: 0; }
    h2 { margin: 0; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .subtitle { margin: 2px 0 0; color: var(--muted); font-size: 13px; }
    .panel {
      background: rgba(255, 255, 255, .86);
      border: 1px solid rgba(216, 224, 231, .95);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(20, 38, 50, .05);
      backdrop-filter: blur(10px);
    }
    .filters {
      padding: 0;
      background: transparent;
      border: 0;
      box-shadow: none;
      backdrop-filter: none;
    }
    .filter-card {
      padding: 14px;
      border: 1px solid rgba(216, 224, 231, .92);
      border-radius: 14px;
      background: rgba(255, 255, 255, .72);
      box-shadow: 0 12px 30px rgba(20, 38, 50, .06);
    }
    .filter-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 900;
    }
    .filter-count {
      color: var(--accent-dark);
      background: var(--soft);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 900;
    }
    label { display: block; margin: 12px 0 6px; color: #344054; font-size: 12px; font-weight: 800; }
    input, select {
      width: 100%;
      border: 1px solid rgba(208, 213, 221, .88);
      border-radius: 12px;
      background: rgba(255,255,255,.92);
      color: var(--ink);
      padding: 11px 12px;
      min-height: 44px;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(11, 127, 115, .14); }
    button {
      border: 0;
      border-radius: 11px;
      background: linear-gradient(135deg, var(--accent), #118f7d);
      color: white;
      min-height: 44px;
      padding: 0 16px;
      font-weight: 850;
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(11, 127, 115, .18);
    }
    button:hover { background: linear-gradient(135deg, var(--accent-dark), var(--accent)); transform: translateY(-1px); }
    button:disabled { opacity: .7; cursor: wait; }
    .secondary {
      background: white;
      color: var(--accent-dark);
      border: 1px solid #a9d8d0;
      box-shadow: none;
    }
    .secondary:hover { background: var(--soft); }
    .filter-actions { display: grid; grid-template-columns: 1fr 88px; gap: 10px; margin-top: 14px; }
    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid rgba(216, 224, 231, .78);
    }
    .examples::before {
      content: "Quick searches";
      flex: 0 0 100%;
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 1px;
    }
    .example {
      background: rgba(227, 243, 239, .88);
      color: #064e46;
      border: 1px solid #c8e4de;
      text-align: center;
      min-height: 34px;
      font-weight: 800;
      box-shadow: none;
      padding: 0 12px;
      font-size: 12px;
      white-space: nowrap;
    }
    .saved-panel {
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid rgba(216, 224, 231, .78);
    }
    .saved-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 10px;
    }
    .saved-clear {
      min-height: 28px;
      padding: 0 9px;
      border-radius: 999px;
      font-size: 11px;
      box-shadow: none;
    }
    .saved-list { display: grid; gap: 8px; }
    .saved-empty {
      color: var(--muted);
      font-size: 13px;
      padding: 10px 0;
    }
    .saved-item {
      border: 1px solid rgba(216, 224, 231, .85);
      border-radius: 12px;
      background: rgba(255,255,255,.68);
      padding: 10px;
      display: grid;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
      cursor: pointer;
      text-align: left;
    }
    .saved-item:hover { border-color: rgba(11, 127, 115, .35); background: rgba(227, 243, 239, .58); }
    .saved-item strong { font-size: 13px; }
    .saved-item span { color: var(--muted); font-size: 12px; }
    .saved-actions { display: flex; gap: 6px; flex-wrap: wrap; }
    .saved-action {
      min-height: 28px;
      padding: 0 9px;
      border-radius: 999px;
      font-size: 11px;
      box-shadow: none;
    }
    .toast {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 20;
      min-width: 260px;
      max-width: 360px;
      padding: 14px 16px;
      border-radius: 14px;
      background: #102a27;
      color: white;
      box-shadow: 0 18px 42px rgba(16, 42, 39, .28);
      font-weight: 750;
      transform: translateY(18px);
      opacity: 0;
      pointer-events: none;
      transition: opacity .18s ease, transform .18s ease;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 190px;
      gap: 14px;
      align-items: stretch;
      padding: 20px;
      margin-bottom: 16px;
      background:
        linear-gradient(135deg, rgba(255,255,255,.95), rgba(255,248,238,.78) 55%, rgba(227,243,239,.72));
      box-shadow: var(--shadow);
    }
    .query-label { font-size: 12px; font-weight: 850; color: var(--muted); margin-bottom: 8px; }
    .query-input {
      min-height: 58px;
      font-size: 16px;
      border-color: rgba(11, 127, 115, .22);
      background: rgba(255, 255, 255, .92);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.8);
    }
    .search-button { min-height: 54px; align-self: end; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric { padding: 16px; position: relative; overflow: hidden; }
    .metric::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 4px;
      background: linear-gradient(var(--accent), var(--coral));
    }
    .metric strong { display: block; font-size: 24px; line-height: 1.1; }
    .metric span { color: var(--muted); font-size: 12px; font-weight: 800; }
    .answer {
      padding: 22px;
      margin-bottom: 16px;
      background:
        linear-gradient(135deg, rgba(255,255,255,.96), rgba(255,248,238,.86));
      box-shadow: var(--shadow);
    }
    .answer h2 { display: flex; align-items: center; gap: 10px; }
    .answer h2::before {
      content: "";
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--coral);
      box-shadow: 0 0 0 5px rgba(189, 75, 61, .12);
    }
    .answer-body { margin-top: 12px; color: #344054; line-height: 1.6; white-space: pre-line; }
    .chat-log { display: grid; gap: 10px; margin-top: 14px; }
    .chat-empty { color: var(--muted); line-height: 1.55; }
    .message {
      max-width: 78%;
      padding: 12px 14px;
      border-radius: 14px;
      line-height: 1.5;
      white-space: pre-line;
      font-size: 14px;
    }
    .message.user {
      justify-self: end;
      background: var(--accent);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .message.assistant {
      justify-self: start;
      background: rgba(255,255,255,.82);
      border: 1px solid rgba(216, 224, 231, .9);
      color: #344054;
      border-bottom-left-radius: 4px;
    }
    .pipeline {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .step {
      padding: 13px;
      border-radius: 14px;
      border: 1px solid rgba(216, 224, 231, .92);
      background: rgba(255,255,255,.74);
      box-shadow: 0 8px 22px rgba(20, 38, 50, .05);
    }
    .step strong { display: block; font-size: 13px; margin-bottom: 4px; }
    .step span { color: var(--muted); font-size: 12px; line-height: 1.35; }
    .feedback-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .results-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 8px 0 12px;
    }
    .results-tools {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 0 0 14px;
    }
    .active-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 32px;
      align-items: center;
    }
    .filter-chip {
      border: 1px solid rgba(11, 127, 115, .20);
      background: rgba(227, 243, 239, .78);
      color: var(--accent-dark);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 850;
    }
    .sort-control {
      width: 178px;
      min-height: 38px;
      padding: 8px 10px;
      font-size: 13px;
      background: rgba(255,255,255,.78);
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
      max-width: 560px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }
    .card {
      padding: 0;
      display: grid;
      gap: 0;
      overflow: hidden;
      border-left: 0;
      background:
        linear-gradient(180deg, rgba(255,255,255,.98), rgba(255,255,255,.90)),
        linear-gradient(135deg, rgba(11,127,115,.10), rgba(255,242,220,.36));
      box-shadow: 0 16px 35px rgba(16, 24, 40, .08);
      transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
    }
    .card:hover {
      transform: translateY(-3px);
      box-shadow: 0 22px 48px rgba(16, 24, 40, .12);
      border-color: rgba(11, 127, 115, .35);
    }
    .card-top {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      padding: 16px 18px 12px;
      border-top: 4px solid transparent;
      background: linear-gradient(#fff, #fff) padding-box, linear-gradient(90deg, var(--accent), var(--coral)) border-box;
    }
    .card-actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .mini-action {
      min-height: 32px;
      padding: 0 11px;
      border-radius: 999px;
      font-size: 12px;
      box-shadow: none;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 850;
    }
    a.mini-action {
      border: 1px solid #a9d8d0;
      color: var(--accent-dark);
      background: rgba(255,255,255,.76);
    }
    .rank-badge {
      min-width: 48px;
      height: 48px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, var(--soft), #fff7e8);
      color: var(--accent-dark);
      font-size: 18px;
      font-weight: 950;
      box-shadow: inset 0 0 0 1px rgba(11, 127, 115, .08);
    }
    .score { color: var(--muted); font-size: 12px; font-weight: 800; margin-top: 4px; text-align: right; }
    .place { margin-top: 5px; color: var(--muted); font-size: 13px; }
    .card-body { padding: 0 18px 16px; display: grid; gap: 12px; }
    .chips { display: flex; gap: 7px; flex-wrap: wrap; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fbfcfd;
      color: #344054;
      font-size: 12px;
      font-weight: 750;
    }
    .rating { border-color: #f3cc7a; background: #fff8e6; color: var(--warn); }
    .tags { color: #344054; line-height: 1.35; }
    .why {
      padding: 12px;
      border: 1px solid #e4e7ec;
      border-radius: 12px;
      background: rgba(248, 250, 251, .78);
      color: #344054;
      font-size: 13px;
      line-height: 1.45;
    }
    .why-title { color: var(--muted); font-size: 11px; font-weight: 900; text-transform: uppercase; margin-bottom: 7px; }
    .why-list { display: grid; gap: 5px; }
    .why-item { display: flex; gap: 7px; align-items: flex-start; }
    .why-dot { width: 6px; height: 6px; border-radius: 999px; background: var(--accent); margin-top: 7px; flex: 0 0 auto; }
    .address { color: var(--muted); font-size: 13px; }
    .popularity { color: var(--muted); font-size: 12px; }
    .empty { padding: 30px; text-align: center; color: var(--muted); }
    @media (max-width: 920px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      main { padding: 16px; }
      .hero { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .status { text-align: left; white-space: normal; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="mark">TW</div>
        <div>
          <h1>TableWise</h1>
          <p class="subtitle">Restaurant discovery assistant</p>
        </div>
      </div>

      <section class="filters">
        <div class="filter-card" style="margin-top: 14px;">
          <div class="filter-title">
            <span>Advanced search</span>
            <span class="filter-count">5 fields</span>
          </div>

          <label for="city">City</label>
          <input id="city" placeholder="Paris" />

          <label for="country">Country</label>
          <input id="country" placeholder="France" />

          <label for="price">Price</label>
          <select id="price">
            <option value="">Any price</option>
            <option value="cheap">Cheap</option>
            <option value="mid">Mid-range</option>
            <option value="expensive">Expensive</option>
          </select>

          <label for="meal">Meal</label>
          <select id="meal">
            <option value="">Any meal</option>
            <option value="breakfast">Breakfast</option>
            <option value="brunch">Brunch</option>
            <option value="lunch">Lunch</option>
            <option value="dinner">Dinner</option>
            <option value="drinks">Drinks</option>
          </select>

          <label for="rating">Minimum rating</label>
          <select id="rating">
            <option value="0">Any rating</option>
            <option value="3.5">3.5+</option>
            <option value="4">4.0+</option>
            <option value="4.5">4.5+</option>
          </select>

          <div class="filter-actions">
            <button id="filterButton" type="button">Apply</button>
            <button id="clearButton" class="secondary" type="button">Clear</button>
          </div>
        </div>
      </section>

      <div class="examples">
        <button class="example" data-q="cheap italian restaurant in rome">Italian Rome</button>
        <button class="example" data-q="vegetarian brunch in paris">Veg brunch</button>
        <button class="example" data-q="seafood dinner in lisbon">Seafood</button>
        <button class="example" data-q="fine dining in milan">Fine dining</button>
      </div>

      <section class="saved-panel">
        <div class="saved-head">
          <span>Your preferences</span>
          <button id="clearSavedButton" class="secondary saved-clear" type="button">Clear</button>
        </div>
        
        <div style="margin-bottom: 20px;">
          <div class="saved-head" style="border-bottom: 1px solid #e4e7ec; padding-bottom: 10px; margin-bottom: 10px;">
            <span>Liked (<span id="likedCount">0</span>)</span>
          </div>
          <div class="saved-list" id="likedList">
            <div class="saved-empty">No liked restaurants yet.</div>
          </div>
        </div>
        
        <div style="margin-bottom: 20px;">
          <div class="saved-head" style="border-bottom: 1px solid #e4e7ec; padding-bottom: 10px; margin-bottom: 10px;">
            <span>Saved (<span id="savedCount">0</span>)</span>
          </div>
          <div class="saved-list" id="savedList">
            <div class="saved-empty">No saved restaurants yet.</div>
          </div>
        </div>
        
        <div>
          <div class="saved-head" style="border-bottom: 1px solid #e4e7ec; padding-bottom: 10px; margin-bottom: 10px;">
            <span>Disliked (<span id="dislikedCount">0</span>)</span>
          </div>
          <div class="saved-list" id="dislikedList">
            <div class="saved-empty">No disliked restaurants yet.</div>
          </div>
        </div>
      </section>
    </aside>

    <main>
      <form class="panel hero" id="searchForm">
        <div>
          <div class="query-label">Ask TableWise</div>
          <input class="query-input" id="q" placeholder="I am visiting Lisbon this weekend. Recommend restaurants near the center." autocomplete="off" />
        </div>
        <button class="search-button" id="searchButton" type="submit">Search</button>
      </form>

      <section class="metrics">
        <div class="panel metric"><strong id="topPicks">0</strong><span>Top picks</span></div>
        <div class="panel metric"><strong id="bestRating">-</strong><span>Best rating</span></div>
        <div class="panel metric"><strong id="priceVibe">Any</strong><span>Price vibe</span></div>
        <div class="panel metric"><strong id="savedCount">0</strong><span>Saved picks</span></div>
      </section>

      <section class="pipeline">
        <div class="step"><strong>1. Parse</strong><span id="pipeParse">Waiting for a request</span></div>
        <div class="step"><strong>2. Retrieve</strong><span id="pipeRetrieve">Restaurant profiles</span></div>
        <div class="step"><strong>3. Rerank</strong><span id="pipeRerank">Quality + constraints</span></div>
        <div class="step"><strong>4. Feedback</strong><span id="pipeFeedback">Ready for human signal</span></div>
      </section>

      <section class="panel answer">
        <h2>TableWise chat</h2>
        <div id="chatLog" class="chat-log">
          <div class="chat-empty">Ask for a restaurant recommendation, then choose a favorite to rerank similar places.</div>
        </div>
      </section>

      <div class="results-head">
        <h2>Ranked restaurants</h2>
        <div class="status" id="status">Ready</div>
      </div>
      <div class="results-tools">
        <div class="active-filters" id="activeFilters"></div>
        <select class="sort-control" id="sortSelect">
          <option value="relevance">Sort by relevance</option>
          <option value="rating">Sort by rating</option>
          <option value="score">Sort by score</option>
          <option value="name">Sort by name</option>
        </select>
      </div>
      <section class="grid" id="results">
        <div class="panel empty">No restaurants loaded yet.</div>
      </section>
    </main>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const $ = (id) => document.getElementById(id);
    const fields = ["q", "city", "country", "price", "meal", "rating"];
    let currentResults = [];
    let savedPicks = JSON.parse(localStorage.getItem("tablewiseSaved") || "[]");
    let likedRestaurants = JSON.parse(localStorage.getItem("tablewiseLiked") || "[]");
    let dislikedRestaurants = JSON.parse(localStorage.getItem("tablewiseDisliked") || "[]");
    let chatMessages = [];
    let feedbackState = {};

    function fmt(value) {
      return Number(value || 0).toLocaleString();
    }

    function ratingValue(result) {
      const parsed = Number.parseFloat(result.rating);
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function samePlace(a, b) {
      return a && b && a.name === b.name && a.city === b.city && a.country === b.country;
    }

    function isSaved(result) {
      return savedPicks.some((item) => samePlace(item, result));
    }

    function isLiked(result) {
      return likedRestaurants.some((item) => samePlace(item, result));
    }

    function isDisliked(result) {
      return dislikedRestaurants.some((item) => samePlace(item, result));
    }

    function saveState() {
      localStorage.setItem("tablewiseSaved", JSON.stringify(savedPicks));
      localStorage.setItem("tablewiseLiked", JSON.stringify(likedRestaurants));
      localStorage.setItem("tablewiseDisliked", JSON.stringify(dislikedRestaurants));
    }

    function showToast(text) {
      $("toast").textContent = text;
      $("toast").classList.add("show");
      window.clearTimeout(showToast.timer);
      showToast.timer = window.setTimeout(() => $("toast").classList.remove("show"), 2400);
    }

    function mapUrl(result) {
      if (result.latitude && result.longitude) {
        return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${result.latitude},${result.longitude}`)}`;
      }
      return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${result.name} ${result.address || result.city}`)}`;
    }

    function renderSaved() {
      $("savedCount").textContent = savedPicks.length;
      $("likedCount").textContent = likedRestaurants.length;
      $("dislikedCount").textContent = dislikedRestaurants.length;
      if (!likedRestaurants.length) {
        $("likedList").innerHTML = `<div class="saved-empty">No liked restaurants yet.</div>`;
      } else {
        $("likedList").innerHTML = likedRestaurants.map((item, index) => `
          <button class="saved-item" type="button" data-list="liked" data-index="${index}">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${escapeHtml(item.city)}, ${escapeHtml(item.country)} - rating ${escapeHtml(item.rating)}</span>
            <div class="saved-actions">
              <span class="saved-action secondary saved-remove" data-list="liked" data-index="${index}">Remove</span>
            </div>
          </button>
        `).join("");
        document.querySelectorAll("#likedList .saved-remove").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            const index = Number(button.dataset.index);
            likedRestaurants.splice(index, 1);
            saveState();
            renderSaved();
            renderResults();
          });
        });
      }
      if (!dislikedRestaurants.length) {
        $("dislikedList").innerHTML = `<div class="saved-empty">No disliked restaurants yet.</div>`;
      } else {
        $("dislikedList").innerHTML = dislikedRestaurants.map((item, index) => `
          <button class="saved-item" type="button" data-list="disliked" data-index="${index}">
            <strong>${escapeHtml(item.name)}</strong>
            <span>${escapeHtml(item.city)}, ${escapeHtml(item.country)} - rating ${escapeHtml(item.rating)}</span>
            <div class="saved-actions">
              <span class="saved-action secondary saved-remove" data-list="disliked" data-index="${index}">Remove</span>
            </div>
          </button>
        `).join("");
        document.querySelectorAll("#dislikedList .saved-remove").forEach((button) => {
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            const index = Number(button.dataset.index);
            dislikedRestaurants.splice(index, 1);
            saveState();
            renderSaved();
            renderResults();
          });
        });
      }
      if (!savedPicks.length) {
        $("savedList").innerHTML = `<div class="saved-empty">No saved restaurants yet.</div>`;
        return;
      }
      $("savedList").innerHTML = savedPicks.map((item, index) => `
        <button class="saved-item" type="button" data-index="${index}">
          <strong>${escapeHtml(item.name)}</strong>
          <span>${escapeHtml(item.city)}, ${escapeHtml(item.country)} - rating ${escapeHtml(item.rating)}</span>
          <div class="saved-actions">
            <span class="saved-action secondary saved-similar" data-index="${index}">Similar</span>
            <span class="saved-action secondary saved-map" data-index="${index}">Map</span>
            <span class="saved-action secondary saved-remove" data-index="${index}">Remove</span>
          </div>
        </button>
      `).join("");
      document.querySelectorAll("#savedList .saved-item").forEach((item) => {
        item.addEventListener("click", () => {
          const result = savedPicks[Number(item.dataset.index)];
          if (result) rerankLike(result);
        });
      });
      document.querySelectorAll("#savedList .saved-similar").forEach((button) => {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          const result = savedPicks[Number(button.dataset.index)];
          if (result) rerankLike(result);
        });
      });
      document.querySelectorAll("#savedList .saved-map").forEach((button) => {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          const result = savedPicks[Number(button.dataset.index)];
          if (result) window.open(mapUrl(result), "_blank", "noreferrer");
        });
      });
      document.querySelectorAll("#savedList .saved-remove").forEach((button) => {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          const index = Number(button.dataset.index);
          savedPicks.splice(index, 1);
          saveState();
          renderSaved();
          renderResults();
        });
      });
    }

    function toggleLike(result) {
      if (isLiked(result)) {
        likedRestaurants = likedRestaurants.filter((item) => !samePlace(item, result));
      } else {
        dislikedRestaurants = dislikedRestaurants.filter((item) => !samePlace(item, result));
        savedPicks = savedPicks.filter((item) => !samePlace(item, result));
        likedRestaurants = [result, ...likedRestaurants];
      }
      saveState();
      renderSaved();
      renderResults();
    }

    function toggleDislike(result) {
      if (isDisliked(result)) {
        dislikedRestaurants = dislikedRestaurants.filter((item) => !samePlace(item, result));
      } else {
        likedRestaurants = likedRestaurants.filter((item) => !samePlace(item, result));
        savedPicks = savedPicks.filter((item) => !samePlace(item, result));
        dislikedRestaurants = [result, ...dislikedRestaurants];
      }
      saveState();
      renderSaved();
      renderResults();
    }

    function toggleSave(result) {
      const exists = isSaved(result);
      if (exists) {
        savedPicks = savedPicks.filter((item) => !(item.name === result.name && item.address === result.address));
      } else {
        savedPicks = [result, ...savedPicks].slice(0, 6);
      }
      saveState();
      renderSaved();
      renderResults();
    }

    function savePick(result) {
      const exists = savedPicks.some((item) => item.name === result.name && item.address === result.address);
      if (!exists) {
        savedPicks = [result, ...savedPicks].slice(0, 6);
        saveState();
        renderSaved();
      }
    }

    function renderActiveFilters(parsed) {
      const chips = [];
      if (parsed.city) chips.push(`City: ${parsed.city}`);
      if (parsed.country) chips.push(`Country: ${parsed.country}`);
      if (parsed.price_bucket) chips.push(`Price: ${parsed.price_bucket}`);
      if (parsed.meal) chips.push(`Meal: ${parsed.meal}`);
      if (parsed.tags && parsed.tags.length) chips.push(`Tags: ${parsed.tags.join(", ")}`);
      $("activeFilters").innerHTML = chips.length
        ? chips.map((chip) => `<span class="filter-chip">${escapeHtml(chip)}</span>`).join("")
        : `<span class="filter-chip">Personalized for your search</span>`;
    }

    function renderChat() {
      if (!chatMessages.length) {
        $("chatLog").innerHTML = `<div class="chat-empty">Ask for a restaurant recommendation, then choose a favorite to rerank similar places.</div>`;
        return;
      }
      $("chatLog").innerHTML = chatMessages.map((message) =>
        `<div class="message ${message.role}">${escapeHtml(message.text)}</div>`
      ).join("");
    }

    function pushChat(role, text) {
      if (!text) return;
      if (chatMessages.length && chatMessages[chatMessages.length - 1].role === role && chatMessages[chatMessages.length - 1].text === text) return;
      chatMessages.push({ role, text });
      chatMessages = chatMessages.slice(-6);
      renderChat();
    }

    function updateFriendlyStats(data) {
      const results = data.results || [];
      const best = results.reduce((max, item) => Math.max(max, ratingValue(item)), 0);
      const prices = results.map((item) => item.price).filter(Boolean);
      const price = prices.length ? prices.sort((a, b) =>
        prices.filter((value) => value === b).length - prices.filter((value) => value === a).length
      )[0] : "Any";
      $("topPicks").textContent = results.length;
      $("bestRating").textContent = best ? best.toFixed(1) : "-";
      $("priceVibe").textContent = price;
      $("savedCount").textContent = savedPicks.length;
    }

    function updatePipeline(data, mode = "search") {
      const parsed = data.parsed || {};
      const parsedText = [parsed.city, parsed.country, parsed.price_bucket, parsed.meal].filter(Boolean).join(" / ") || "Open-ended request";
      $("pipeParse").textContent = mode === "preferences" ? "Loading your preferences" : (mode === "more-like-preferences" ? "Analyzing your liked restaurants" : parsedText);
      $("pipeRetrieve").textContent = mode === "preferences" ? `${data.results.length} liked restaurants found` : (mode === "more-like-preferences" ? `${data.results.length} similar restaurants found` : `${data.results.length} recommendations surfaced`);
      $("pipeRerank").textContent = mode === "preferences" ? "Sorted by: cuisine, price, location" : (mode === "more-like-preferences" ? "Sorted by: cuisine, price, location match" : (mode === "similar" ? "Reranked from favorite" : "Ranked by match quality"));
      $("pipeFeedback").textContent = mode === "preferences" || mode === "more-like-preferences" ? "Mark as good or not right" : "Good / Not right buttons enabled";
    }

    function sortedResults() {
      const mode = $("sortSelect").value;
      const results = currentResults.filter((item) => !isDisliked(item));
      return [...results].sort((a, b) => {
        if (mode === "rating") return ratingValue(b) - ratingValue(a) || b.score - a.score;
        if (mode === "score" || mode === "relevance") return b.score - a.score;
        if (mode === "name") return a.name.localeCompare(b.name);
        return a.rank - b.rank;
      });
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function setBusy(isBusy) {
      $("searchButton").disabled = isBusy;
      $("filterButton").disabled = isBusy;
      $("searchButton").textContent = isBusy ? "Searching..." : "Search";
      $("filterButton").textContent = isBusy ? "Applying..." : "Apply filters";
      if (isBusy) $("status").textContent = "Ranking candidates...";
    }

    function card(result, index) {
      const saved = isSaved(result);
      const liked = isLiked(result);
      const disliked = isDisliked(result);
      const feedback = feedbackState[`${result.name}|${result.address}`];
      const reasons = result.reasons.map((reason) => `
        <div class="why-item">
          <span class="why-dot"></span>
          <span>${escapeHtml(reason)}</span>
        </div>
      `).join("");
      return `
        <article class="panel card">
          <div class="card-top">
            <div>
              <h3>${escapeHtml(result.name)}</h3>
              <div class="place">${escapeHtml(result.city)}, ${escapeHtml(result.country)}</div>
            </div>
            <div>
              <div class="rank-badge">#${result.rank}</div>
              <div class="score">${result.score}</div>
            </div>
          </div>
          <div class="card-body">
            <div class="chips">
              <span class="chip rating">Rating ${escapeHtml(result.rating)}</span>
              <span class="chip">${escapeHtml(result.price)}</span>
              <span class="chip">${escapeHtml(result.meals)}</span>
            </div>
            <div class="tags">${escapeHtml(result.tags)}</div>
            <div class="why">
              <div class="why-title">Why this matches</div>
              <div class="why-list">${reasons}</div>
            </div>
            <div class="address">${escapeHtml(result.address || result.popularity)}</div>
            <div class="popularity">${escapeHtml(result.popularity)}</div>
            <div class="card-actions">
              <button class="mini-action save-action" type="button" data-index="${index}">${saved ? "Saved" : "Save"}</button>
              <button class="mini-action like-action" type="button" data-index="${index}">${liked ? "Liked" : "Good"}</button>
              <button class="mini-action dislike-action" type="button" data-index="${index}">${disliked ? "Disliked" : "Not right"}</button>
              <button class="mini-action similar-action" type="button" data-index="${index}">More like this</button>
              <a class="mini-action" href="${mapUrl(result)}" target="_blank" rel="noreferrer">Map</a>
            </div>
          </div>
        </article>
      `;
    }

    function renderResults() {
      const results = sortedResults();
      $("results").innerHTML = results.length
        ? results.map((result, index) => card(result, index)).join("")
        : `<div class="panel empty">No matching restaurants. Relax one filter and try again.</div>`;
      document.querySelectorAll(".save-action").forEach((button) => {
        button.addEventListener("click", () => {
          const result = sortedResults()[Number(button.dataset.index)];
          if (result) toggleSave(result);
        });
      });
      document.querySelectorAll(".like-action").forEach((button) => {
        button.addEventListener("click", () => {
          const result = sortedResults()[Number(button.dataset.index)];
          if (result) toggleLike(result);
        });
      });
      document.querySelectorAll(".dislike-action").forEach((button) => {
        button.addEventListener("click", () => {
          const result = sortedResults()[Number(button.dataset.index)];
          if (result) toggleDislike(result);
        });
      });
      document.querySelectorAll(".similar-action").forEach((button) => {
        button.addEventListener("click", () => {
          const result = sortedResults()[Number(button.dataset.index)];
          if (result) rerankLike(result);
        });
      });
    }

    async function sendFeedback(result, feedback) {
      feedbackState[`${result.name}|${result.address}`] = feedback;
      renderResults();
      $("pipeFeedback").textContent = feedback === "good" ? `${result.name} marked as a good match` : `${result.name} marked as not right`;
      $("status").textContent = feedback === "good" ? `You liked ${result.name}` : `Feedback saved for ${result.name}`;
      showToast(feedback === "good" ? `You liked ${result.name}` : `Feedback saved for ${result.name}`);
      const params = new URLSearchParams({
        query: $("q").value.trim(),
        restaurant: result.name,
        city: result.city,
        rating: result.rating,
        feedback
      });
      try {
        await fetch(`/api/feedback?${params.toString()}`);
        pushChat("assistant", feedback === "good"
          ? `Great, I noted ${result.name} as a restaurant you like.`
          : `Thanks, I marked ${result.name} as not a good match.`
        );
      } catch (error) {
        pushChat("assistant", "I could not save that feedback locally.");
      }
    }

    async function rerankLike(result) {
      setBusy(true);
      pushChat("user", `Show me more restaurants like ${result.name}.`);
      const params = new URLSearchParams({
        name: result.name,
        city: result.city,
        country: result.country,
        price: result.price,
        tags: result.tags,
        meals: result.meals,
        limit: "9"
      });
      try {
        const response = await fetch(`/api/similar?${params.toString()}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        currentResults = data.results;
        updateFriendlyStats(data);
        updatePipeline(data, "similar");
        renderActiveFilters(data.parsed);
        renderResults();
        pushChat("assistant", data.answer);
        $("status").textContent = `Reranked from ${result.name}`;
      } catch (error) {
        pushChat("assistant", error.message);
      } finally {
        setBusy(false);
      }
    }

    async function search(clearFilters = false, addChat = true) {
      if (clearFilters) {
        fields.forEach((id) => {
          if (id === "q") return;
          $(id).value = id === "rating" ? "0" : "";
        });
      }
      
      const query = $("q").value.trim().toLowerCase();
      
      // Check if user is asking for preferences-based search
      if (query.includes("show me") && (query.includes("according to my preferences") || query.includes("my preferences"))) {
        return await searchPreferences();
      }
      
      // Check if user is asking for more like their preferences
      if ((query.includes("show me more") || query.includes("give me more") || query.includes("find more")) && (query.includes("like") || query.includes("similar"))) {
        return await searchMoreLikePreferences();
      }
      
      const started = performance.now();
      setBusy(true);
      const params = new URLSearchParams();
      fields.forEach((id) => {
        const value = $(id).value.trim();
        if (value) params.set(id, value);
      });
      params.set("limit", "9");
      try {
        const response = await fetch(`/api/search?${params.toString()}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        currentResults = data.results;
        updateFriendlyStats(data);
        updatePipeline(data, "search");
        renderActiveFilters(data.parsed);
        renderResults();
        if (addChat) {
          pushChat("user", data.query || $("q").value.trim() || "Find restaurants");
        }
        pushChat("assistant", data.answer);
        const filterText = [data.parsed.city, data.parsed.country, data.parsed.price_bucket, data.parsed.meal].filter(Boolean).join(" / ");
        $("status").textContent = data.relaxed ? `Showing relaxed matches for ${filterText || "your query"}` : `Showing matches for ${filterText || "your query"}`;
      } catch (error) {
        $("status").textContent = error.message;
        $("results").innerHTML = `<div class="panel empty">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    async function searchPreferences() {
      setBusy(true);
      try {
        const params = new URLSearchParams({
          liked: JSON.stringify(likedRestaurants)
        });
        const response = await fetch(`/api/preferences?${params.toString()}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        currentResults = data.results;
        updateFriendlyStats(data);
        updatePipeline(data, "preferences");
        renderActiveFilters(data.parsed);
        renderResults();
        pushChat("user", $("q").value.trim());
        pushChat("assistant", data.answer);
        $("status").textContent = `Showing your ${data.results.length} liked restaurants, sorted by cuisine, price, and location`;
      } catch (error) {
        pushChat("assistant", error.message);
        $("status").textContent = error.message;
        $("results").innerHTML = `<div class="panel empty">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    async function searchMoreLikePreferences() {
      setBusy(true);
      try {
        const params = new URLSearchParams({
          liked: JSON.stringify(likedRestaurants)
        });
        const response = await fetch(`/api/more-like-preferences?${params.toString()}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        currentResults = data.results;
        updateFriendlyStats(data);
        updatePipeline(data, "more-like-preferences");
        renderActiveFilters(data.parsed);
        renderResults();
        pushChat("user", $("q").value.trim());
        pushChat("assistant", data.answer);
        $("status").textContent = `Found ${data.results.length} new restaurants similar to your preferences`;
      } catch (error) {
        pushChat("assistant", error.message);
        $("status").textContent = error.message;
        $("results").innerHTML = `<div class="panel empty">${escapeHtml(error.message)}</div>`;
      } finally {
        setBusy(false);
      }
    }

    $("searchForm").addEventListener("submit", (event) => {
      event.preventDefault();
      search(true, true);
    });

    $("filterButton").addEventListener("click", () => search(false, false));
    
    $("clearButton").addEventListener("click", () => {
      fields.forEach((id) => $(id).value = id === "rating" ? "0" : "");
      chatMessages = [];
      renderChat();
      $("results").innerHTML = `<div class="panel empty">No restaurants loaded yet.</div>`;
      $("topPicks").textContent = "0";
      $("bestRating").textContent = "-";
      $("priceVibe").textContent = "Any";
      $("savedCount").textContent = savedPicks.length;
      $("status").textContent = "Ready";
      currentResults = [];
      $("activeFilters").innerHTML = "";
      $("pipeParse").textContent = "Waiting for a request";
      $("pipeRetrieve").textContent = "Restaurant profiles";
      $("pipeRerank").textContent = "Quality + constraints";
      $("pipeFeedback").textContent = "Ready for human signal";
    });

    $("sortSelect").addEventListener("change", renderResults);
    $("clearSavedButton").addEventListener("click", () => {
      savedPicks = [];
      saveState();
      renderSaved();
      renderResults();
    });

    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        $("q").value = button.dataset.q;
        search(true, true);
      });
    });

    renderSaved();
    renderChat();
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
        self.send_json({"error": "Not found"}, status=404)


def main() -> None:
    port = 8501
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"TableWise UI running at http://127.0.0.1:{port}")
    print(f"Dataset: {DATA_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
