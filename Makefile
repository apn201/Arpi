# ARPI - build targets. Windows: run from Git Bash.

PY := python

.PHONY: help test eval eval-known sheets singles serve models manifest synth deploy secrets clean

help:
	@echo "test        pytest, offline, no AWS"
	@echo "eval        synthetic sweep vs OpenCV's decoder (slow, ~3 min)"
	@echo "eval-known  tears and occlusions with a 1000-code company list"
	@echo "sheets      physical set, whole sheets, with the cascade"
	@echo "singles     physical set, one code per crop (the normal use case)"
	@echo "serve       the scanner page and API on localhost:8013"
	@echo "models      fetch the text model, checked against its hash"
	@echo "manifest    checksum the physical photos"
	@echo "synth       cdk synth"
	@echo "deploy      build the image and deploy (aws sso login --profile whyf first)"
	@echo "secrets     scan every commit before the repo goes public"

test:
	$(PY) -m pytest

eval:
	$(PY) tools/evaluate.py --n 8 --out build/eval.json

eval-known:
	$(PY) tools/evaluate.py --n 8 --kinds tear,occlusion --known 1000 --out build/eval_known.json

sheets:
	$(PY) tools/eval_sheets.py data/physical/photos --cascade --out build/sheets.json

singles:
	$(PY) tools/eval_singles.py data/physical/photos --out build/singles.json

serve:
	$(PY) tools/serve_local.py

models:
	$(PY) tools/fetch_models.py

manifest:
	$(PY) tools/photo_manifest.py

synth:
	cd infra && npx cdk synth --quiet

deploy: models
	cd infra && npx cdk deploy --profile $${AWS_PROFILE:-whyf}

secrets:
	$(PY) tools/check_secrets.py

clean:
	rm -rf build .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
