#!/bin/bash

export OLLAMA_BASE_DIR=/mnt/local/disk2/ollama
export OLLAMA_MODELS=$OLLAMA_BASE_DIR/models
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OLLAMA_BIN="$OLLAMA_BASE_DIR/bin/ollama"
COMMAND="$OLLAMA_BIN serve"

gpus='all'
hostport="localhost:12343"

export CUDA_VISIBLE_DEVICES="all"
export OLLAMA_HOST="$hostport"

OLLAMA_HOST="$hostport" $OLLAMA_BIN run gemma3:4b
