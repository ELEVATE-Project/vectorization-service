# ai-vector-service

FastAPI service that ingests documents (PDF, DOCX, XLSX, CSV, TXT/Markdown or a URL) into Qdrant and
searches them with hybrid search: multi-field dense embeddings plus BM25 keyword vectors.

## Setup

For setup and run instructions, see the [Qdrant Developer Guide](https://github.com/ELEVATE-Project/commons-backend/blob/main/docs/integrations/vector_db/qdrant/developer_guide.md).

## API documentation (Swagger)

FastAPI generates the Swagger docs from the code, so they always match the running service.

| | Local | Other environments |
|---|---|---|
| Swagger UI | http://localhost:8000/docs | `https://<host>/vector/docs` |
| ReDoc | http://localhost:8000/redoc | `https://<host>/vector/redoc` |
| OpenAPI JSON | http://localhost:8000/openapi.json | `https://<host>/vector/openapi.json` |

Generate the OpenAPI (Swagger) file:

```bash
# From a running service
curl -s http://localhost:8000/openapi.json -o openapi.json

# Without starting the server (run from the repo root with the virtual environment active)
QDRANT_CHECK_COMPATIBILITY=false python -c "import json; from app.main import app; json.dump(app.openapi(), open('openapi.json', 'w'), indent=2)"
```

The file can be imported into Postman or opened in Swagger Editor.

## Postman collection

[postman/vectorization-service.postman_collection.json](postman/vectorization-service.postman_collection.json)
has every endpoint, with one request per success and validation case and a saved example response for each.

1. In Postman, choose **Import** and select the file.
2. Set the collection variables: `baseUrl` (`http://localhost:8000` locally, `https://<host>/vector` elsewhere),
   `sourceId`, `companyId` and `theme`.
3. For upload, update and upsert requests, choose a file in the `file` field. Each request's description says
   which kind of file to use.

Run the folders in this order to get the same responses as the examples: upload, search and utilities, update,
upsert, metadata, theme, delete. The collection description lists known issues in the current behavior.

## Developer guide

- [Qdrant Developer Guide](https://github.com/ELEVATE-Project/commons-backend/blob/main/docs/integrations/vector_db/qdrant/developer_guide.md)
