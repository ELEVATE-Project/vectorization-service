from fastapi.responses import JSONResponse
from typing import Any
import json
import math
from fastapi import FastAPI


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special float values"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj):
                return "NaN"
            elif math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
        return super().default(obj)


class CustomJSONResponse(JSONResponse):
    """Custom JSON response class using the custom encoder"""

    def render(self, content: Any) -> bytes:
        # First, clean the content to replace NaN/Inf with None
        cleaned_content = self._clean_floats(content)
        return json.dumps(
            cleaned_content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")
    
    def _clean_floats(self, obj: Any) -> Any:
        """Recursively clean NaN and Inf values from nested structures"""
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: self._clean_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_floats(item) for item in obj]
        return obj


def setup_custom_json_handling(app: FastAPI) -> None:
    """Configure FastAPI to use custom JSON response handling"""
    app.json_encoder = CustomJSONEncoder

    # Override the default JSONResponse with our custom one
    @app.middleware("http")
    async def custom_json_middleware(request, call_next):
        response = await call_next(request)
        if isinstance(response, JSONResponse) and not isinstance(response, CustomJSONResponse):
            return CustomJSONResponse(
                content=response.body.decode(),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        return response
