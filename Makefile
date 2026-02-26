.PHONY: setup install data train demo clean test

setup:
	mkdir -p data/{raw,processed,final,selected,adapters}
	cp .env.example .env
	@echo "Edit .env with your credentials"

install:
	pip install -r requirements.txt

data:
	python -m src.data.loader

train:
	@echo "Training Coder agent..."
	python -m src.training.trainer --agent coder
	@echo "Training Reviewer agent..."
	python -m src.training.trainer --agent reviewer

demo:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf data/processed/* data/final/* data/selected/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

test:
	pytest tests/ -v