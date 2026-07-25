#!/usr/bin/env bash
# eval/collect_all_facts.sh
folder=$1
if [ -z "$folder" ]; then
  echo "Usage: collect_all_facts.sh <folder>"
  exit 1
fi
pushd "$folder" > /dev/null
cat train.txt facts.txt valid.txt test.txt > all.txt
popd > /dev/null
echo "Created $folder/all.txt"
