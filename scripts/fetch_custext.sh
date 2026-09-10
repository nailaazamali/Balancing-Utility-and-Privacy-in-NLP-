#!/usr/bin/env bash
set -euo pipefail
mkdir -p external
if [ -d external/CusText/.git ]; then
  echo "external/CusText already exists"
  git -C external/CusText rev-parse HEAD
  exit 0
fi

git clone https://github.com/sai4july/CusText.git external/CusText
printf "Fetched CusText commit: "
git -C external/CusText rev-parse HEAD
echo "Record this commit with any new experiment; this release does not claim it is the historical paper-baseline commit."
