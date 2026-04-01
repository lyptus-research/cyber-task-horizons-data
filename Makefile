.PHONY: setup figures clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and pull LFS files
	uv sync
	git lfs pull

figures:  ## Reproduce all paper figures (from pre-built Stage 1 data)
	cd analysis && uv run dvc repro

clean:  ## Remove generated figures
	rm -rf analysis/figures/out/*.png
