.PHONY: clean lint format requirements upgrade build sync_models_up sync_models_down test register_environment preflight

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = docdistance
MODULE_NAME = docdistance
PYTHON_VERSION = 3.13
# python3 first: Ubuntu/Debian ship no bare `python` unless python-is-python3 is installed,
# which made the default goal and preflight fail with a misleading "install Python" message.
# Resolve to a bare NAME, never a path: an empty value would make preflight's `command -v`
# run with no operand, which exits 0 and silently disables the guard, and an absolute path
# word-splits unquoted on a prefix containing a space.
PYTHON_INTERPRETER ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)
RUFF_VERSION = 0.16.2

# Set SKIP_VERSION_INCREMENT=1 to skip auto-bumping the patch version in install/build
SKIP_VERSION_INCREMENT ?= 0

#################################################################################
# STYLES                                                                        #
#################################################################################

MSG_PREFIX = \033[1m\033[36m>>>\033[0m
WARN_PREFIX = \033[33m>>>\033[0m
ERR_PREFIX = \033[31m>>>\033[0m
WARN_STYLE = \033[33m
ERR_STYLE = \033[31m
HIGHLIGHT_STYLE = \033[1m\033[94m
OK_STYLE = \033[92m
NO_STYLE = \033[0m

#################################################################################
# ENVIRONMENT CONFIGURATION                                                     #
#################################################################################

# unified environment name for all managers
ENV_NAME = docdistance
# uv configuration
VENV_PATH = $(PROJECT_DIR)/.venv
UV_OPTS =

#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Install Python dependencies
.PHONY: requirements
requirements:
	@echo "$(MSG_PREFIX) installing requirements with uv"
	uv $(UV_OPTS) sync --python $(PROJECT_DIR)/.venv --extra dev
## Upgrade Python dependencies to latest versions
.PHONY: upgrade
upgrade:
	@echo "$(MSG_PREFIX) upgrading packages with uv"
	uv $(UV_OPTS) sync --python $(PROJECT_DIR)/.venv --extra dev --upgrade

## Delete all compiled Python files
clean:
	@echo "$(MSG_PREFIX) removing cache and compiled files"
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name '__pycache__' -prune -exec rm -r {} +
	@find . -type d -name '*.egg-info' -prune -exec rm -r {} +
	@find . -type d -name '.ipynb_checkpoints' -prune -exec rm -r {} +
	@find . -type d -name '.pytest_cache' -prune -exec rm -r {} +
	@echo "$(MSG_PREFIX) removing dist and build directory"
	@rm -rf build dist

## Restore .env from encrypted .env.enc (or create empty)
.env:
	@if [ -f ".env.enc" ]; then \
		echo "$(MSG_PREFIX) decrypting .env.enc"; \
		openssl enc -d -aes-256-cbc -pbkdf2 -in .env.enc -out .env || { rm -f .env; echo "$(ERR_PREFIX) $(ERR_STYLE)decryption failed$(NO_STYLE)"; exit 1; }; \
	else \
		echo "$(MSG_PREFIX) creating empty .env"; \
		touch .env; \
	fi

## Encrypt .env to .env.enc (AES-256)
.env.enc: .env
	@echo "$(MSG_PREFIX) encrypting .env"
	@openssl enc -aes-256-cbc -pbkdf2 -in .env -out .env.enc
	@echo "$(OK_STYLE)>>> .env.enc file successfully created$(NO_STYLE)"

## Lint using ruff (use `make format` to do formatting)
# ruff is pinned: unpinned uvx resolves the latest release, and a release that
# widens the default rule set turns CI red on source that never changed
lint:
	@echo "$(MSG_PREFIX) linting the sourcecode"
	uvx ruff@$(RUFF_VERSION) format --check
	uvx ruff@$(RUFF_VERSION) check

## Format source code with ruff
format:
	@echo "$(MSG_PREFIX) formatting the sourcecode"
	uvx ruff@$(RUFF_VERSION) check --fix
	uvx ruff@$(RUFF_VERSION) format
## Run tests
test:
	@echo "$(MSG_PREFIX) checking for tests"
	@$(PROJECT_DIR)/.venv/bin/pytest --collect-only ./tests > /dev/null 2>&1; RESULT="$$?"; \
	if [ "$$RESULT" != "5" ]; then \
		echo "$(MSG_PREFIX) executing python tests"; \
		$(PROJECT_DIR)/.venv/bin/pytest --cov -v ./tests; \
	else \
		echo "$(WARN_PREFIX) $(WARN_STYLE)WARNING: no tests present$(NO_STYLE)"; \
	fi
#################################################################################
# UV ENVIRONMENT MANAGEMENT                                                     #
#################################################################################

## Preflight check for required tools
preflight:
	@if ! command -v $(PYTHON_INTERPRETER) >/dev/null 2>&1; then \
		echo "$(ERR_PREFIX) $(ERR_STYLE)ERROR: $(PYTHON_INTERPRETER) not found$(NO_STYLE)"; \
		echo "$(ERR_PREFIX) $(ERR_STYLE)install Python from https://www.python.org/downloads/$(NO_STYLE)"; \
		exit 1; \
	fi

## Set up Python interpreter environment
create_environment: preflight
	@if [ -d "$(PROJECT_DIR)/.venv" ]; then \
		echo "$(MSG_PREFIX) virtual environment already exists at $(HIGHLIGHT_STYLE).venv$(NO_STYLE). Skipping creation."; \
	else \
		if ! command -v uv >/dev/null 2>&1; then \
			echo "$(MSG_PREFIX) installing uv"; \
			pip install -q uv; \
		fi; \
		echo "$(MSG_PREFIX) creating uv virtual environment"; \
		uv $(UV_OPTS) venv -q --python $(PYTHON_VERSION); \
		echo "$(MSG_PREFIX) new uv virtual environment created. Activate with:"; \
		echo "$(MSG_PREFIX) Windows: $(HIGHLIGHT_STYLE).\\\.venv\\\Scripts\\\activate$(NO_STYLE)"; \
		echo "$(MSG_PREFIX) Unix/macOS: $(HIGHLIGHT_STYLE)source ./.venv/bin/activate$(NO_STYLE)"; \
		echo "$(MSG_PREFIX) installing dependencies"; \
		uv $(UV_OPTS) pip install -q --python $(PROJECT_DIR)/.venv -e ".[dev]"; \
		if command -v nb_venv_kernels >/dev/null 2>&1; then \
			echo "$(MSG_PREFIX) registering Jupyter kernel for $(HIGHLIGHT_STYLE)$(ENV_NAME)$(NO_STYLE)"; \
			nb_venv_kernels register --name $(ENV_NAME) $(PROJECT_DIR)/.venv >/dev/null 2>&1; \
			echo "$(OK_STYLE)>>> Kernel registered successfully$(NO_STYLE)"; \
		else \
			echo "$(MSG_PREFIX) registering Jupyter kernel with ipykernel"; \
			$(PROJECT_DIR)/.venv/bin/python -m ipykernel install --user --name=$(ENV_NAME) --display-name "Python [uv env:$(ENV_NAME)]"; \
			echo "$(OK_STYLE)>>> Kernel registered as $(ENV_NAME)$(NO_STYLE)"; \
		fi; \
	fi

## Remove previously created environment
remove_environment:
	@echo "$(MSG_PREFIX) removing uv virtual environment at $(HIGHLIGHT_STYLE).venv$(NO_STYLE)"
	@echo "$(MSG_PREFIX) unregistering Jupyter kernel $(HIGHLIGHT_STYLE)$(ENV_NAME)$(NO_STYLE)"
	@if command -v nb_venv_kernels >/dev/null 2>&1; then \
		nb_venv_kernels unregister $(PROJECT_DIR)/.venv >/dev/null 2>&1 || true; \
	else \
		jupyter kernelspec uninstall -y $(ENV_NAME) >/dev/null 2>&1 || true; \
	fi
	@-rm -rf ~/.local/share/jupyter/kernels/$(ENV_NAME) 2>/dev/null || true
	@rm -rf $(PROJECT_DIR)/.venv
	@echo "$(OK_STYLE)>>> Environment removed$(NO_STYLE)"
## Register Jupyter kernel for the environment
register_environment:
	@if [ ! -d "$(PROJECT_DIR)/.venv" ]; then \
		echo "$(ERR_PREFIX) $(ERR_STYLE)ERROR: uv virtual environment not found at .venv. Run 'make create_environment' first$(NO_STYLE)"; \
		exit 1; \
	fi
	@if command -v nb_venv_kernels >/dev/null 2>&1; then \
		echo "$(MSG_PREFIX) registering Jupyter kernel for $(HIGHLIGHT_STYLE)$(ENV_NAME)$(NO_STYLE)"; \
		nb_venv_kernels register --name $(ENV_NAME) $(PROJECT_DIR)/.venv >/dev/null 2>&1; \
		echo "$(OK_STYLE)>>> Kernel registered successfully$(NO_STYLE)"; \
	else \
		echo "$(MSG_PREFIX) registering Jupyter kernel with ipykernel"; \
		$(PROJECT_DIR)/.venv/bin/python -m ipykernel install --user --name=$(ENV_NAME) --display-name "Python [uv env:$(ENV_NAME)]"; \
		echo "$(OK_STYLE)>>> Kernel registered as $(ENV_NAME)$(NO_STYLE)"; \
	fi

## Install src modules (editable)
install: create_environment requirements clean increment_version_number .env

	@echo "$(MSG_PREFIX) installing $(MODULE_NAME) in editable mode"
	@uv $(UV_OPTS) pip install -q --python $(PROJECT_DIR)/.venv -e .
	@echo "$(OK_STYLE)>>> $(MODULE_NAME) installed$(NO_STYLE)"

## Build package
build: clean install test
	@echo "$(MSG_PREFIX) building $(MODULE_NAME)"
	$(PROJECT_DIR)/.venv/bin/python -m build --wheel

## Publish to PyPI with twine (full gate: build -> install -> test, then upload)
.PHONY: publish
publish: build install test
	@echo "$(MSG_PREFIX) building sdist for $(MODULE_NAME)"
	$(PROJECT_DIR)/.venv/bin/python -m build --sdist
	@echo "$(MSG_PREFIX) validating distributions with twine"
	$(PROJECT_DIR)/.venv/bin/python -m twine check dist/*
	@echo "$(MSG_PREFIX) uploading $(MODULE_NAME) to PyPI with twine"
	$(PROJECT_DIR)/.venv/bin/python -m twine upload dist/*

## Increment build number (skip with SKIP_VERSION_INCREMENT=1)
increment_version_number:
	@if [ "$(SKIP_VERSION_INCREMENT)" = "1" ]; then \
		echo "$(MSG_PREFIX) skipping version increment (SKIP_VERSION_INCREMENT=1)"; \
	else \
		echo "$(MSG_PREFIX) incrementing build number"; \
		$(PROJECT_DIR)/.venv/bin/python -c "import re; c=open('pyproject.toml').read(); m=re.search(r'version = \"(\\d+)\\.(\\d+)\\.(\\d+)\"',c); v=f'{m[1]}.{m[2]}.{int(m[3])+1}'; c=re.sub(r'version = \"\\d+\\.\\d+\\.\\d+\"',f'version = \"{v}\"',c,count=1); open('pyproject.toml','w').write(c); print('New version:',v)"; \
	fi

#################################################################################
# S3 MODEL MIRROR                                                               #
#################################################################################
# The published OpenVINO INT8 IRs that `docdistance init --source s3://...` pulls. Layout under the
# bucket matches the model registry subdirs, so init resolves each model by name.
S3_PROFILE ?= stellars-tech
S3_MODELS_URI ?= s3://general-purpose/docdistance
S3_MODEL_DIRS = mmbert-openvino-int8 reranker-openvino-int8 nli-openvino-int8 sat-3l-sm

## Upload the model IRs in models/<registry-subdir> to the S3 mirror (stellars-tech profile)
sync_models_up:
	@for d in $(S3_MODEL_DIRS); do \
		if [ -d "$(PROJECT_DIR)/models/$$d" ]; then \
			echo "$(MSG_PREFIX) syncing models/$$d -> $(S3_MODELS_URI)/$$d"; \
			aws s3 sync "$(PROJECT_DIR)/models/$$d" "$(S3_MODELS_URI)/$$d" --profile $(S3_PROFILE); \
		else \
			echo "$(WARN_PREFIX) $(WARN_STYLE)skip models/$$d (not present)$(NO_STYLE)"; \
		fi; \
	done

## Download the model IRs from the S3 mirror into models/<registry-subdir>
sync_models_down:
	@for d in $(S3_MODEL_DIRS); do \
		echo "$(MSG_PREFIX) syncing $(S3_MODELS_URI)/$$d -> models/$$d"; \
		aws s3 sync "$(S3_MODELS_URI)/$$d" "$(PROJECT_DIR)/models/$$d" --profile $(S3_PROFILE); \
	done

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = sys.stdin.read(); \
matches = re.findall(r'\n## ([^\n]+)\n(?!\.PHONY)([a-zA-Z_.][a-zA-Z0-9_.-]*):', lines); \
matches = sorted(matches, key=lambda x: x[1].lower()); \
print('\nAvailable rules:\n'); \
print('\n'.join(['\033[36m{:25}\033[0m{}'.format(*reversed(match)) for match in matches])); \
print()
endef
export PRINT_HELP_PYSCRIPT

## Print the list of available commands
help:
	@$(PYTHON_INTERPRETER) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

# EOF
