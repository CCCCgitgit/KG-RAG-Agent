# KG-RAG Agent development and deployment shortcuts.
#
# Common usage:
#   make install
#   make test
#   make run QUERY="UNITED STATES 与 CHINA 有什么关系？"
#   make api-dev
#   make docker-up

SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= python
PIP := $(PYTHON) -m pip
NPM ?= npm
FRONTEND_DIR ?= frontend

PROFILE ?= demo
QUERY ?=
CONFIG ?=
INPUT ?= data/demo/examples/demo_questions.json
OUTPUT_DIR ?= outputs/evaluation
LIMIT ?=

HOST ?= 127.0.0.1
PORT ?= 8000
WORKERS ?= 1
LOG_LEVEL ?= info

CONFIG_ARG = $(if $(strip $(CONFIG)),--config "$(CONFIG)",)
QUERY_ARG = $(if $(strip $(QUERY)),-q "$(QUERY)",)
LIMIT_ARG = $(if $(strip $(LIMIT)),--limit "$(LIMIT)",)

.PHONY: \
	help env init-dirs \
	install install-core install-api install-retrieval install-dev \
	test test-unit test-smoke test-integration test-e2e test-api test-memory \
	coverage collect \
	lint lint-fix format format-check typecheck check \
	build build-wheel clean \
	run interactive api api-dev \
	frontend-install frontend-dev frontend-build frontend-preview \
	build-data build-data-demo build-data-production build-data-no-vector migrate-data \
	evaluate final-check \
	docker-config docker-build docker-up docker-down docker-restart \
	docker-logs docker-ps docker-shell

help: ## Show available commands.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env: ## Create .env from .env.example when .env does not exist.
	@$(PYTHON) -c "from pathlib import Path; import shutil; src=Path('.env.example'); dst=Path('.env'); print('Keeping existing .env' if dst.exists() else 'Created .env from .env.example'); None if dst.exists() else shutil.copyfile(src, dst)"

init-dirs: ## Create local data, log, cache, evaluation and Memory directories.
	@$(PYTHON) -c "from pathlib import Path; paths=('data/demo/kg','data/demo/processed','data/demo/vector_store','data/demo/examples','data/production/raw','outputs/evaluation','outputs/memory','outputs/cache','logs'); [Path(p).mkdir(parents=True, exist_ok=True) for p in paths]; print('Created runtime directories.')"

install: ## Install the complete editable runtime from requirements.txt.
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-core: ## Install only the core editable package.
	$(PIP) install -e .

install-api: ## Install the package with FastAPI and Uvicorn.
	$(PIP) install -e ".[api]"

install-retrieval: ## Install the package with vector retrieval dependencies.
	$(PIP) install -e ".[retrieval]"

install-dev: ## Install development, testing, API and retrieval dependencies.
	$(PIP) install -e ".[dev,test,api,retrieval]"

test: ## Run the complete test suite.
	$(PYTHON) -m pytest

test-unit: ## Run unit tests.
	$(PYTHON) -m pytest -m unit

test-smoke: ## Run smoke tests.
	$(PYTHON) -m pytest -m smoke

test-integration: ## Run integration tests.
	$(PYTHON) -m pytest -m integration

test-e2e: ## Run end-to-end tests.
	$(PYTHON) -m pytest -m e2e

test-api: ## Run FastAPI tests.
	$(PYTHON) -m pytest -m api

test-memory: ## Run Memory isolation and persistence tests.
	$(PYTHON) -m pytest -m memory

coverage: ## Run tests with branch coverage.
	$(PYTHON) -m pytest \
		--cov=kg_rag_agent \
		--cov-branch \
		--cov-report=term-missing \
		--cov-report=html

collect: ## Validate test discovery without executing tests.
	$(PYTHON) -m pytest --collect-only -q

lint: ## Run Ruff checks without modifying files.
	$(PYTHON) -m ruff check src tests scripts

lint-fix: ## Apply safe Ruff fixes.
	$(PYTHON) -m ruff check --fix src tests scripts

format: ## Format source, tests and scripts with Ruff.
	$(PYTHON) -m ruff format src tests scripts

format-check: ## Check formatting without modifying files.
	$(PYTHON) -m ruff format --check src tests scripts

typecheck: ## Run Mypy on the installed package.
	$(PYTHON) -m mypy -p kg_rag_agent

check: lint format-check typecheck test ## Run the full local quality gate.

build: clean ## Build source distribution and wheel.
	$(PYTHON) -m build

build-wheel: clean ## Build only the wheel.
	$(PYTHON) -m build --wheel

clean: ## Remove caches and build/test artifacts; keep data and Memory.
	@$(PYTHON) -c "from pathlib import Path; import shutil; roots=(Path('.'),); dirs={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','htmlcov','build','dist','.eggs'}; [shutil.rmtree(p, ignore_errors=True) for root in roots for p in root.rglob('*') if p.is_dir() and p.name in dirs]; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').glob('*.egg-info')]; [p.unlink(missing_ok=True) for name in ('.coverage','coverage.xml','pytest-report.xml','junit.xml') for p in (Path(name),)]; print('Removed local build and test caches.')"

run: ## Ask one question. Example: make run QUERY="your question".
	$(PYTHON) -m kg_rag_agent $(QUERY_ARG) $(CONFIG_ARG)

interactive: ## Start a multi-turn CLI session with Memory continuity.
	$(PYTHON) -m kg_rag_agent --interactive $(CONFIG_ARG) --memory-status

api: ## Start the FastAPI service.
	$(PYTHON) -m kg_rag_agent \
		--serve \
		--host "$(HOST)" \
		--port "$(PORT)" \
		--workers "$(WORKERS)" \
		--log-level "$(LOG_LEVEL)" \
		$(CONFIG_ARG)

api-dev: ## Start FastAPI with reload for local development.
	$(PYTHON) -m kg_rag_agent \
		--serve \
		--reload \
		--host "$(HOST)" \
		--port "$(PORT)" \
		--workers 1 \
		--log-level "$(LOG_LEVEL)" \
		$(CONFIG_ARG)


frontend-install: ## Install frontend dependencies.
	cd $(FRONTEND_DIR) && $(NPM) ci

frontend-dev: ## Start the Vue development server.
	cd $(FRONTEND_DIR) && $(NPM) run dev

frontend-build: ## Type-check and build the Vue frontend.
	cd $(FRONTEND_DIR) && $(NPM) run build

frontend-preview: ## Preview the built frontend.
	cd $(FRONTEND_DIR) && $(NPM) run preview

build-data: init-dirs ## Build offline artifacts using PROFILE=demo or production.
	$(PYTHON) scripts/build_all.py --profile "$(PROFILE)"

build-data-demo: init-dirs ## Build demo graph and vector-store artifacts.
	$(PYTHON) scripts/build_all.py --profile demo

build-data-production: init-dirs ## Build production graph and vector-store artifacts.
	$(PYTHON) scripts/build_all.py --profile production

build-data-no-vector: init-dirs ## Build graph artifacts without a vector store.
	$(PYTHON) scripts/build_all.py \
		--profile "$(PROFILE)" \
		--skip-vector-store

migrate-data: ## Copy legacy data into the Production profile.
	$(PYTHON) scripts/migrate_legacy_data.py

evaluate: init-dirs ## Evaluate the formal AgentService pipeline.
	$(PYTHON) scripts/evaluate.py \
		--input "$(INPUT)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(LIMIT_ARG) \
		$(CONFIG_ARG)

docker-config: ## Validate and render the Docker Compose configuration.
	docker compose config

docker-build: ## Build the API container image.
	docker compose build

docker-up: init-dirs ## Build and start the API container in the background.
	docker compose up --build -d

docker-down: ## Stop and remove Compose containers and network.
	docker compose down

docker-restart: ## Restart the API service.
	docker compose restart api

docker-logs: ## Follow API container logs.
	docker compose logs -f --tail=200 api

docker-ps: ## Show Compose service status.
	docker compose ps

docker-shell: ## Open a shell inside the running API container.
	docker compose exec api /bin/sh

final-check: test frontend-build docker-config ## Run final backend, frontend and Compose checks.
