.PHONY: run clean

run:
	@export JAX_PLATFORMS=cpu ; uv run setup.py

clean:
	@rm -rf */__pycache__/
	@rm -rf __pycache__/
	@rm -rf output/
