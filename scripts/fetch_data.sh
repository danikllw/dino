#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
curl -fsSL 'https://raw.githubusercontent.com/jpospinalo/MachineLearning/main/nlp/dinos.csv' -o data/dinos.csv
echo 'Downloaded data/dinos.csv'
