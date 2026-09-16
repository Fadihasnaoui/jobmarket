.PHONY: help up down api frontend test lint build

help:
	@echo Available commands:
	@echo   make up       - start the complete application with Docker
	@echo   make down     - stop Docker services
	@echo   make api      - run the FastAPI backend locally
	@echo   make frontend - run the React frontend locally
	@echo   make test     - run backend unit tests
	@echo   make lint     - check Python code quality
	@echo   make build    - build the frontend for production

up:
	docker compose up --build

down:
	docker compose down

api:
	python -m uvicorn jobmarket.api.main:app --reload --port 8123

frontend:
	cd frontend && npm run dev

test:
	python -m pytest tests/unit -q

lint:
	python -m ruff check src tests

build:
	cd frontend && npm run build
