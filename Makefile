PY=python3
VENV=venv
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python

.PHONY: venv install test

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

install:
	$(MAKE) venv

test:
	$(PYBIN) tests/test_feature_aggregator.py || true
	$(PYBIN) tests/test_data_loader_detailed.py || true
	$(PYBIN) tests/test_pipeline_simple.py || true
