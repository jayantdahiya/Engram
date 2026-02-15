# Repository Guidelines

## Project Structure & Module Organization
- `engram-backend/`: primary FastAPI service and memory system.
- `engram-backend/api/`, `engram-backend/services/`, `engram-backend/models/`, `engram-backend/core/`, `engram-backend/tasks/`: API layer, business logic, schemas, shared config, and async jobs.
- `engram-backend/infrastructure/docker/`: Dockerfiles, compose stack, monitoring, and init scripts.
- `engram-backend/tests/`: backend unit and integration tests.
- `engram-mcp/`: MCP server package that connects assistants to Engram.
- `engram-mcp/src/engram_mcp/`: MCP server/client/tools implementation; top-level `test_*.py` files cover MCP behavior.
- `tests/`: repository-level integration and cross-component tests (`tests/backend/`, `tests/mcp/`).
- `examples/`: runnable API usage examples.

## Build, Test, and Development Commands
- `docker-compose -f engram-backend/infrastructure/docker/docker-compose.yml up -d`: start full local stack.
- `cd engram-backend && python -m api.main`: run API locally without Docker rebuild.
- `cd engram-backend && pytest tests/ -v --tb=short`: run backend tests.
- `pytest tests/ -v`: run root cross-component test suite.
- `cd engram-mcp && uv sync && ./start-mcp.sh`: install MCP deps and start MCP server.
- `ruff check engram-backend/ engram-mcp/`: lint Python code.
- `black --check engram-backend/ && isort --check-only engram-backend/`: enforce formatting/import order.

## Coding Style & Naming Conventions
- Use Python 3.11+, 4-space indentation, and type hints for new/changed functions.
- Follow Black + isort conventions (Black line length 88; isort profile `black`).
- Ruff is enabled for correctness/import/upgrade rules; fix warnings before PR.
- Naming: `snake_case` for files/functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.

## Testing Guidelines
- Framework: `pytest` with async support (`pytest-asyncio`).
- Test naming is enforced in `tests/pytest.ini`: files `test_*.py`, classes `Test*`, functions `test_*`.
- Use markers for intent: `unit`, `integration`, `slow`.
- Prefer targeted runs while developing, then run full suites before opening a PR.

## Commit & Pull Request Guidelines
- Match existing history style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:` + concise imperative summary.
- Keep commits focused by concern (API, retrieval, MCP, infra).
- PRs should include: what changed, why, test commands run, and any env/config updates.
- Ensure CI passes (`ruff`, `black`, `isort`, tests, Docker build, Bandit) before requesting review.

## Security & Configuration Tips
- Start from `engram-backend/env.example`; never commit secrets or populated `.env` files.
- Use non-default credentials for local shared environments and all production deployments.
