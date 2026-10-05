# Utils Modules

**Purpose.** Read this page for an overview of the three helper modules in `app/utils/`: JSON serialization, language helpers and query preprocessing. Consumers that own the detailed behaviour are linked per section.

| Module | Used by | Status |
|---|---|---|
| `json_handler.py` | `app/main.py` (`default_response_class`, `EmbeddingError` handler) | active |
| `language_utils.py` | `translation_service.py` (imports only), `query_service.py` | translation disabled; effectively inert |
| `query_preprocessor.py` | `PrioritizedSearchService` | active |

## 1. `json_handler.py`

- `CustomJSONEncoder(json.JSONEncoder)`: `default()` maps NaN to `"NaN"` and +-inf to `"Infinity"`/`"-Infinity"`, but `default()` is only called for objects the encoder cannot serialize, and floats are natively serializable, so this branch never fires.
- `CustomJSONResponse(JSONResponse)`:
  - `render(content)` calls `_clean_floats` then `json.dumps(cleaned, cls=CustomJSONEncoder, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":"))` and encodes UTF-8.
  - `_clean_floats(obj)` recursively replaces NaN/Inf with `None` inside floats, dicts, lists and tuples.
- `setup_custom_json_handling(app)`: sets `app.json_encoder` and adds an HTTP middleware that re-wraps plain `JSONResponse` objects. **Never called**; `main.py` instead uses `default_response_class=CustomJSONResponse`. If it were enabled, its `content=response.body.decode()` would double-encode JSON as a string.

Net behaviour: all API JSON is compact, UTF-8 (no `\uXXXX` escapes) and NaN-safe (`null`). Qdrant scores that become NaN after normalization therefore appear as `null`.

## 2. `language_utils.py`

- Module import configures logging globally (`logging.basicConfig` with `FileHandler('app.log')` and `StreamHandler`). Side effect: creates `app.log` in the CWD.
- `TranslationError(message, status_code, should_retry=False)` custom exception.
- `exponential_backoff(retry_count, base_delay=1.0)`: `min(base_delay * 2**retry_count, 60)`.
- `_handle_rate_limit` (sleeps `Retry-After`, default 60 s), `_handle_server_error`, `_handle_timeout`: retry helpers returning the incremented counter; server-error raises `TranslationError` after `max_retries`.
- `translate_text(text, source_language, target_language, api_key, max_retries=3)`: POSTs an AI4Bharat/ULCA-style payload to `settings.TRANSLATION_API_URL` (`https://demo-api.models.ai4bharat.org/inference/translation/v2`), 30 s timeout, returns `response.json()['output'][0]['target']`. 429 and 5xx and timeouts retry with backoff.
- `detect_language(text)`: returns `'hi'` if the text shares any character with a hard-coded set of Devanagari letters, else `'en'`.

Notes: retries use blocking `time.sleep` (would block the event loop if ever called from async code); the final `raise HTTPException(... last_error)` references `last_error`, which is never assigned beyond `None`. In the current system `translation_service.process_chunk` returns `(chunk_text, False)` without calling these functions, and `QueryService._process_query_language` hard-codes English, so the module is effectively unused. Ingestion-side behaviour belongs to [Ingestion pipeline](../services/ingestion/pipeline.md).

## 3. `query_preprocessor.py`

Used by the search service to normalize the **embedding** query (the original query is retained for keyword matching). Search-side usage is documented in [Search overview](../services/search/overview.md).

| Function | Behaviour |
|---|---|
| `_load_spacy_model()` | singleton `spacy.load("en_core_web_sm")`; raises `RuntimeError` with install hint if missing |
| `preprocess_query(query)` | see below |
| `is_spacy_model_available()` | `spacy.util.is_package("en_core_web_sm")`, without loading |

`preprocess_query` rules:

1. Empty/whitespace: returns `""`.
2. If `len(words) < settings.SHORT_QUERY_THRESHOLD` (default 3) **or** `len(stripped) < 20`: returns `stripped.lower()` without loading spaCy.
3. Otherwise runs spaCy and drops tokens with `pos_ == "PRON"`, `is_stop`, `is_punct`, `is_space`; keeps `token.text.lower()`.
4. If nothing remains, returns `stripped.lower()`.
5. On any non-`RuntimeError` exception returns the original `stripped` (not lowercased); `RuntimeError` (model load failure) propagates.

Note the short-query test is an OR: a 2-word query of 25 characters is treated as short, and a 5-word query under 20 characters too.

Known issues:

- Module docstring promises lemmatization; the code deliberately uses `token.text` (comment says "no lemmatization").
- The first long query pays the spaCy model load (lazy singleton) latency.
- The error branch returns the original-case string whereas other branches lowercase; harmless for the embedding model but inconsistent.

## Related pages

- [App lifecycle](app_lifecycle.md)
- [Clients](clients.md)
- [Search overview](../services/search/overview.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Auxiliary services](../services/auxiliary_services.md)
