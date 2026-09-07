PYTHON ?= uv run python

.PHONY: install api test demo check-amd build-llama download-model benchmark
install:
	uv sync
api:
	uv sync --extra api
test:
	$(PYTHON) -m unittest discover -s tests -v
demo:
	uv run lynx demo
check-amd:
	./scripts/check-amd.sh
build-llama:
	./scripts/build-llama.sh
download-model:
	./scripts/download-model.sh
benchmark:
	./scripts/bench-llama.sh
