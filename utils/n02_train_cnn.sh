#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-train_cnn}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-train_cnn.txt}"

EMBEDDINGS_PATH=""
METADATA_PATH=""
MIN_CLASS_COUNT="${MIN_CLASS_COUNT:-50}"
TEST_SIZE="${TEST_SIZE:-0.2}"
RANDOM_STATE="${RANDOM_STATE:-42}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --embeddings-path PATH --metadata-path PATH [options]

Required:
  --embeddings-path PATH     Path to embeddings CSV
  --metadata-path PATH       Path to metadata JSON

Optional:
  --env-name NAME            Micromamba env name (default: ${ENV_NAME})
  --python-version VERSION   Python version for env (default: ${PYTHON_VERSION})
  --requirements FILE        Requirements file (default: ${REQUIREMENTS_FILE})
  --min-class-count N        Default: ${MIN_CLASS_COUNT}
  --test-size FLOAT          Default: ${TEST_SIZE}
  --random-state N           Default: ${RANDOM_STATE}
  --epochs N                 Default: ${EPOCHS}
  --batch-size N             Default: ${BATCH_SIZE}
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="$2"
      shift 2
      ;;
    --embeddings-path)
      EMBEDDINGS_PATH="$2"
      shift 2
      ;;
    --metadata-path)
      METADATA_PATH="$2"
      shift 2
      ;;
    --min-class-count)
      MIN_CLASS_COUNT="$2"
      shift 2
      ;;
    --test-size)
      TEST_SIZE="$2"
      shift 2
      ;;
    --random-state)
      RANDOM_STATE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${EMBEDDINGS_PATH}" || -z "${METADATA_PATH}" ]]; then
  echo "Error: --embeddings-path and --metadata-path are required." >&2
  usage
  exit 1
fi

echo "Creating micromamba environment: ${ENV_NAME}"
micromamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip

echo "Installing dependencies from ${REQUIREMENTS_FILE}"
micromamba run -n "${ENV_NAME}" python -m pip install -r "${REQUIREMENTS_FILE}"

echo "Starting training"
micromamba run -n "${ENV_NAME}" python -m scripts.n16_train_kmer_cnn \
  --embeddings-path "${EMBEDDINGS_PATH}" \
  --metadata-path "${METADATA_PATH}" \
  --min-class-count "${MIN_CLASS_COUNT}" \
  --test-size "${TEST_SIZE}" \
  --random-state "${RANDOM_STATE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}"