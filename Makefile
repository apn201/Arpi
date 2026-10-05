# ARPI - build targets. Windows: run from Git Bash.

PY := python
BUNDLE ?= $${ARPI_BUNDLE:-build/lambda}

.PHONY: help test eval eval-known bundle synth deploy secrets clean

help:
	@echo "test        pytest, offline, no AWS"
	@echo "eval        synthetic sweep vs OpenCV's decoder (slow, ~3 min)"
	@echo "eval-known  tears and occlusions with a 1000-code company list"
	@echo "bundle      lambda zip contents, no Docker"
	@echo "synth       cdk synth"
	@echo "deploy      cdk deploy (AWS_PROFILE=whyf, same account as WhyF)"
	@echo "secrets     scan every commit before the repo goes public"

test:
	$(PY) -m pytest

eval:
	$(PY) tools/evaluate.py --n 8 --out build/eval.json

eval-known:
	$(PY) tools/evaluate.py --n 8 --kinds tear,occlusion --known 1000 --out build/eval_known.json

bundle:
	$(PY) tools/build_lambda.py --out $(BUNDLE)

synth:
	cd infra && ARPI_BUNDLE=$(BUNDLE) npx cdk synth --quiet

deploy:
	cd infra && ARPI_BUNDLE=$(BUNDLE) npx cdk deploy --profile $${AWS_PROFILE:-whyf}

secrets:
	$(PY) tools/check_secrets.py

clean:
	rm -rf build .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
