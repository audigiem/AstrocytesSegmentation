# Makefile for AstroCa project
# 3D+time fluorescence image astrocyte segmentation with feature extraction

# Variables
DOCS_DIR = docs
RUNS_DIR = runs
TESTS_DIR = tests
DOXYFILE = Doxyfile

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  install           - Install dependencies using Poetry"
	@echo "  run               - Run the main processing pipeline"
	@echo "  test              - Run component tests"
	@echo "  profile           - Run with line-by-line profiling"
	@echo "  visualize         - Run visualization of segmentation results"
	@echo "  doc               - Generate and open documentation"
	@echo "  clean             - Remove all generated files"
	@echo "  clean-docs        - Remove documentation files only"
	@echo "  clean-runs        - Remove run output files only"
	@echo "  architecture      - Show project architecture"

# Installation
.PHONY: install
install:
	@echo "Installing dependencies with Poetry..."
	poetry install

# Main execution targets
.PHONY: run
run:
	poetry run python $(TESTS_DIR)/main.py

.PHONY: run-stats
run-stats:
	poetry run python $(TESTS_DIR)/main.py --stats

.PHONY: run-memstats
run-memstats:
	@echo "Warning: Memory profiling significantly increases computation time"
	poetry run python $(TESTS_DIR)/main.py --memstats

# Testing
.PHONY: test
test:
	@echo "Running component tests..."
	poetry run python -m pytest $(TESTS_DIR)/componentTest/ -v

# Profiling
.PHONY: profile
profile:
	@echo "Running line-by-line profiling..."
	@echo "Note: Add @profile decorator to functions you want to profile"
	poetry run kernprof -l -v $(TESTS_DIR)/main.py

# Visualization
.PHONY: visualize
visualize:
	poetry run python $(TESTS_DIR)/visualizationSegmentation.py

# Documentation
.PHONY: doc
doc: check-doxygen
	@echo "Generating documentation..."
	doxygen $(DOXYFILE)
	@echo "Opening documentation..."
	xdg-open $(DOCS_DIR)/html/index.html

.PHONY: doc-latex
doc-latex: doc
	@echo "Compiling LaTeX documentation..."
	cd $(DOCS_DIR)/latex && make all

.PHONY: check-doxygen
check-doxygen:
	@which doxygen > /dev/null || (echo "Error: Doxygen not found. Install with: sudo apt install doxygen" && exit 1)

# Architecture
.PHONY: architecture
architecture:
	@if [ -f "./showArchitectureProject.sh" ]; then \
		chmod +x ./showArchitectureProject.sh && ./showArchitectureProject.sh; \
	else \
		echo "Architecture script not found"; \
	fi

# Cleaning targets
.PHONY: clean
clean: clean-docs clean-runs
	@echo "All generated files removed"

.PHONY: clean-docs
clean-docs:
	@echo "Removing documentation files..."
	@if [ -d "$(DOCS_DIR)" ]; then \
		cd $(DOCS_DIR) && rm -rf *; \
	fi

.PHONY: clean-runs
clean-runs:
	@echo "Removing run output files..."
	@if [ -d "$(RUNS_DIR)" ]; then \
		cd $(RUNS_DIR) && rm -rf *; \
	fi

# Development targets
.PHONY: env-info
env-info:
	@echo "Poetry environment information:"
	poetry env info
	poetry env list

.PHONY: deps-update
deps-update:
	@echo "Updating dependencies..."
	poetry update

.PHONY: lint
lint:
	@echo "Running code linting..."
	poetry run black --check astroca/ $(TESTS_DIR)/

.PHONY: format
format:
	@echo "Formatting code..."
	poetry run black astroca/ $(TESTS_DIR)/

# Legacy aliases for backward compatibility
.PHONY: segmentations time_profiling visualization
segmentations: run-stats
time_profiling: profile
visualization: visualize