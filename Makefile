PYTHON ?= python

.PHONY: test lint ingest ask

test:
	$(PYTHON) -m pytest -m "not integration"

lint:
	$(PYTHON) -m ruff check .

ingest:
	$(PYTHON) -m app.cli ingest

ask:
	$(PYTHON) -m app.cli ask "$(QUESTION)"
