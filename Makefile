# makefile for executing and cleaning up the project

TARGET_DIR_RUNs = runs


clean_doc:
	cd docs && rm -rf *

clean:
	cd $(TARGET_DIR_RUNs) && rm -rf *
	cd docs && rm -rf *

clean_runs:
	cd $(TARGET_DIR_RUNs) && rm -rf *

segmentations:
	poetry run python tests/main.py --stats

architecture:
	./showArchitectureProject.sh

