PYTHON ?= python

.PHONY: test lint ingest ask transcribe structure eval eval-asr eval-retrieval ui api load-test-1 load-test-3

test:
	$(PYTHON) -m pytest -m "not integration"

lint:
	$(PYTHON) -m ruff check .

ingest:
	$(PYTHON) -m app.cli ingest

ask:
	$(PYTHON) -m app.cli ask "$(QUESTION)"

transcribe:
	$(PYTHON) -m app.cli transcribe "$(AUDIO)"

structure:
	$(PYTHON) -m app.cli structure "$(TEXT_FILE)"

eval: eval-asr eval-retrieval

eval-asr:
	$(PYTHON) -m eval.eval_asr

eval-retrieval:
	$(PYTHON) -m eval.eval_retrieval

ui:
	$(PYTHON) -m app.ui.gradio_app

api:
	$(PYTHON) -m uvicorn app.api.main:app --reload

load-test-1:
	$(PYTHON) scripts/load_test.py --requests 20 --concurrency 1 --no-generate

load-test-3:
	$(PYTHON) scripts/load_test.py --requests 60 --concurrency 3 --no-generate
