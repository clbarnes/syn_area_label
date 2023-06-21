.PHONY: format
format:
	isort . \
	&& black .

.PHONY: lint
lint:
	black --check .
	isort --check .
	flake8 .

lint-type: lint
	mypy .

.PHONY: test
test:
	pytest --verbose

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt \
	&& pip install -e ".[dev]"

.PHONY: container
container:
	sudo apptainer build --bind "$(shell pwd)/.git:/project/.git" syn_area_label.sif ./Apptainer
