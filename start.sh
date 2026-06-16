#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# Vectorization Service - Start Script
# Works for fresh installation AND restarts (fully idempotent)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Validate .env ─────────────────────────────────────────────────────────────
[[ -f ".env" ]] || error ".env file not found in $PROJECT_DIR"
set -a; source .env; set +a
info "Loaded .env"

# ── Parse DB connection from POSTGRES_DATABASE_URI ────────────────────────────
DB_URI="${POSTGRES_DATABASE_URI:-}"
[[ -z "$DB_URI" ]] && error "POSTGRES_DATABASE_URI not set in .env"

DB_USER=$(echo "$DB_URI" | sed -E 's|postgresql://([^:]+):.*|\1|')
DB_PASS=$(echo "$DB_URI" | sed -E 's|postgresql://[^:]+:([^@]+)@.*|\1|')
DB_NAME=$(echo "$DB_URI" | sed -E 's|.*/([^/]+)$|\1|')

# ── 1. PostgreSQL ─────────────────────────────────────────────────────────────
info "Checking PostgreSQL..."
if pg_isready -q 2>/dev/null; then
  info "PostgreSQL is already running"
else
  info "Starting PostgreSQL..."
  brew services start postgresql@14 2>/dev/null || brew services start postgresql 2>/dev/null
  sleep 3
  pg_isready -q || { sleep 3; pg_isready || error "PostgreSQL failed to start."; }
fi
info "PostgreSQL is ready"

psql postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1 || \
  psql postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';" >/dev/null

psql postgres -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 || \
  psql postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" >/dev/null

psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" >/dev/null
info "Database '$DB_NAME' is ready"

# ── 2. Redis ──────────────────────────────────────────────────────────────────
info "Checking Redis..."
if redis-cli ping &>/dev/null; then
  info "Redis is already running"
else
  info "Starting Redis..."
  brew services start redis
  sleep 2
  redis-cli ping | grep -q PONG || error "Redis failed to start."
fi
info "Redis is ready"

# ── 3. Qdrant (Docker) ────────────────────────────────────────────────────────
QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_CONTAINER="qdrant"

info "Checking Qdrant..."
if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
    info "Restarting stopped Qdrant container..."
    docker start "$QDRANT_CONTAINER"
  else
    info "Starting Qdrant container..."
    docker run -d \
      --name "$QDRANT_CONTAINER" \
      -p "${QDRANT_PORT}:6333" \
      -p "6334:6334" \
      -v "$PROJECT_DIR/.qdrant_storage:/qdrant/storage" \
      qdrant/qdrant
  fi

  info "Waiting for Qdrant to be ready..."
  for i in $(seq 1 15); do
    curl -sf "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" &>/dev/null && break
    sleep 2
  done
fi
curl -sf "http://${QDRANT_HOST}:${QDRANT_PORT}/healthz" &>/dev/null \
  || error "Qdrant is not responding at http://${QDRANT_HOST}:${QDRANT_PORT}"
info "Qdrant is ready"

# ── 4. Virtual environment ────────────────────────────────────────────────────
info "Checking virtual environment..."

# Detect stale venv (absolute paths break when project folder is moved)
if [[ -d ".venv" ]] && ! .venv/bin/python -c "import sys" &>/dev/null; then
  warn ".venv is broken (stale paths). Recreating..."
  rm -rf .venv
fi

if [[ ! -d ".venv" ]]; then
  info "Creating virtual environment..."
  uv venv
fi

# ── 5. Install dependencies ───────────────────────────────────────────────────
info "Installing dependencies..."
uv pip install -r requirements.txt

info "Ensuring spaCy English model is present..."
.venv/bin/python -m spacy download en_core_web_sm

# ── 6. Start the app ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  All services up — starting vectorization API   ${NC}"
echo -e "${GREEN}  Swagger UI: http://localhost:8000/docs          ${NC}"
echo -e "${GREEN}  Qdrant UI:  http://localhost:${QDRANT_PORT}/dashboard    ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
