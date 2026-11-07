#!/usr/bin/env bash

set -euo pipefail

RUN_EVALUATION=false
SKIP_COVERAGE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-evaluation)
            RUN_EVALUATION=true
            ;;
        --skip-coverage)
            SKIP_COVERAGE=true
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac

    shift
done

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

PROJECT_ROOT="$(
    cd "${SCRIPT_DIR}/.." &&
    pwd
)"

run_step() {
    local description="$1"
    shift

    echo
    echo "========================================"
    echo "${description}"
    echo "========================================"

    "$@"
}

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

run_step \
    "Checking Python runtime" \
    python -c "
import sys

print('Python:', sys.version)
print('Executable:', sys.executable)

assert sys.version_info[:2] == (3, 11), (
    'Python 3.11 is required'
)
"

run_step \
    "Checking installed dependencies" \
    python -m pip check

run_step \
    "Testing native package imports" \
    python -X faulthandler -c "
import numpy
print('numpy', numpy.__version__)

import scipy
print('scipy', scipy.__version__)

import sklearn
print('sklearn', sklearn.__version__)

import torch
print('torch', torch.__version__)

import sentence_transformers
print(
    'sentence-transformers',
    sentence_transformers.__version__,
)

import chromadb
print('chromadb', chromadb.__version__)

import fastapi
print('fastapi', fastapi.__version__)

import prometheus_client
print('prometheus-client imported')

print('NATIVE IMPORT CHECK PASSED')
"

run_step \
    "Checking application imports" \
    python -c "
from src.ingest import normalize_ticket
from src.classify import TicketClassifier
from src.retrieve import ChromaRetriever
from src.route import decide_route
from src.generate import ResponseGenerator
from src.guardrails import GuardrailEngine
from src.workflow import CustomerSuccessWorkflow
from src.api import app
from src.evaluate import evaluate_run
from evaluation.harness import run_evaluation_harness

print('APPLICATION IMPORT CHECK PASSED')
"

run_step \
    "Validating classifier artifact" \
    python -c "
from pathlib import Path
from src.classify import TicketClassifier

path = Path('artifacts/ticket_classifier.joblib')

assert path.exists(), (
    'Classifier artifact is missing'
)

TicketClassifier.load(path)

print('CLASSIFIER ARTIFACT PASSED')
"

run_step \
    "Validating documentation corpus" \
    python -c "
from src.retrieve import load_and_chunk_corpus

passages = load_and_chunk_corpus(
    'data/documentation.json'
)

assert passages, (
    'Documentation produced no passages'
)

print('Documentation passages:', len(passages))
print('DOCUMENTATION CHECK PASSED')
"

if [[ "${SKIP_COVERAGE}" == "true" ]]; then
    run_step \
        "Running complete pytest suite" \
        python -m pytest -q
else
    run_step \
        "Running complete pytest suite with coverage" \
        python -m pytest \
            --cov=src \
            --cov=evaluation \
            --cov-report=term-missing \
            --cov-report=xml \
            -q
fi

if [[ "${RUN_EVALUATION}" == "true" ]]; then
    EVALUATION_COMMAND=(
        python
        -m
        evaluation.harness
    )

    run_step \
        "Running real validation evaluation" \
        "${EVALUATION_COMMAND[@]}"
else
    echo
    echo "Real evaluation skipped."
    echo "Use --run-evaluation to enable it."
fi

echo
echo "========================================"
echo "ALL VALIDATION CHECKS PASSED"
echo "========================================"
