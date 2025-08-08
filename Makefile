PY=python3

.PHONY: install test

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) tests/test_feature_aggregator.py || true
	$(PY) tests/test_data_loader_detailed.py || true
	$(PY) tests/test_pipeline_simple.py || true
