[CmdletBinding()]
param(
    [switch]$RunEvaluation,
    [switch]$SkipCoverage
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path `
    -Parent `
    $PSScriptRoot

$originalLocation = Get-Location

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "========================================"
    Write-Host $Description
    Write-Host "========================================"

    & python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw (
            "$Description failed with exit code " +
            "$LASTEXITCODE."
        )
    }
}

try {
    Set-Location -LiteralPath $projectRoot

    Write-Host "Project root: $projectRoot"

    Invoke-PythonStep `
        -Description "Checking Python runtime" `
        -Arguments @(
            "-c",
            (
                "import sys; " +
                "print('Python:', sys.version); " +
                "print('Executable:', sys.executable); " +
                "assert sys.version_info[:2] == (3, 11), " +
                "'Python 3.11 is required'"
            )
        )

    Invoke-PythonStep `
        -Description "Checking installed dependencies" `
        -Arguments @(
            "-m",
            "pip",
            "check"
        )

    Invoke-PythonStep `
        -Description "Testing native package imports" `
        -Arguments @(
            "-X",
            "faulthandler",
            "-c",
            (
                "import numpy; " +
                "print('numpy', numpy.__version__); " +
                "import scipy; " +
                "print('scipy', scipy.__version__); " +
                "import sklearn; " +
                "print('sklearn', sklearn.__version__); " +
                "import torch; " +
                "print('torch', torch.__version__); " +
                "import sentence_transformers; " +
                "print(" +
                "'sentence-transformers', " +
                "sentence_transformers.__version__" +
                "); " +
                "import chromadb; " +
                "print('chromadb', chromadb.__version__); " +
                "import fastapi; " +
                "print('fastapi', fastapi.__version__); " +
                "import prometheus_client; " +
                "print('prometheus-client imported'); " +
                "print('NATIVE IMPORT CHECK PASSED')"
            )
        )

    Invoke-PythonStep `
        -Description "Checking application imports" `
        -Arguments @(
            "-c",
            (
                "from src.ingest import normalize_ticket; " +
                "from src.classify import TicketClassifier; " +
                "from src.retrieve import ChromaRetriever; " +
                "from src.route import decide_route; " +
                "from src.generate import ResponseGenerator; " +
                "from src.guardrails import GuardrailEngine; " +
                "from src.workflow import CustomerSuccessWorkflow; " +
                "from src.api import app; " +
                "from src.evaluate import evaluate_run; " +
                "from evaluation.harness import " +
                "run_evaluation_harness; " +
                "print('APPLICATION IMPORT CHECK PASSED')"
            )
        )

    Invoke-PythonStep `
        -Description "Validating classifier artifact" `
        -Arguments @(
            "-c",
            (
                "from pathlib import Path; " +
                "from src.classify import TicketClassifier; " +
                "path=Path('artifacts/ticket_classifier.joblib'); " +
                "assert path.exists(), " +
                "'Classifier artifact is missing'; " +
                "TicketClassifier.load(path); " +
                "print('CLASSIFIER ARTIFACT PASSED')"
            )
        )

    Invoke-PythonStep `
        -Description "Validating documentation corpus" `
        -Arguments @(
            "-c",
            (
                "from src.retrieve import " +
                "load_and_chunk_corpus; " +
                "passages=load_and_chunk_corpus(" +
                "'data/documentation.json'" +
                "); " +
                "assert passages, " +
                "'Documentation produced no passages'; " +
                "print('Documentation passages:', " +
                "len(passages)); " +
                "print('DOCUMENTATION CHECK PASSED')"
            )
        )

    if ($SkipCoverage) {
        Invoke-PythonStep `
            -Description "Running complete pytest suite" `
            -Arguments @(
                "-m",
                "pytest",
                "-q"
            )
    }
    else {
        Invoke-PythonStep `
            -Description (
                "Running complete pytest suite " +
                "with coverage"
            ) `
            -Arguments @(
                "-m",
                "pytest",
                "--cov=src",
                "--cov=evaluation",
                "--cov-report=term-missing",
                "--cov-report=xml",
                "-q"
            )
    }

    if ($RunEvaluation) {
        $evaluationArguments = @(
            "-m",
            "evaluation.harness"
        )

        Invoke-PythonStep `
            -Description (
                "Running real validation evaluation"
            ) `
            -Arguments $evaluationArguments
    }
    else {
        Write-Host ""
        Write-Host (
            "Real evaluation skipped. Use " +
            "-RunEvaluation to enable it."
        )
    }

    Write-Host ""
    Write-Host "========================================"
    Write-Host "ALL VALIDATION CHECKS PASSED"
    Write-Host "========================================"
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Set-Location -LiteralPath $originalLocation
}
