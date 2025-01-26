.PHONY: run clean

run:
	@uv run setup.py

clean:
	@rm -rf */__pycache__/
	@rm -rf __pycache__/
	@rm -rf output/
