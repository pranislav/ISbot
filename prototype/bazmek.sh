#!/bin/bash

export OLLAMA_BASE_DIR=/mnt/local/disk2/ollama
export OLLAMA_MODELS=$OLLAMA_BASE_DIR/models
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OLLAMA_BIN="$OLLAMA_BASE_DIR/bin/ollama"
COMMAND="$OLLAMA_BIN serve"

gpus=''
hostport=''

while [ -n "$1" ]; do
    case $1 in
        --help|-h)
            echo "Usage: $0  [devices]  [host:port]"
            echo "   devices   - gpus for CUDA_VISIBLE_DEVICES. e.g. '11,12'"
            echo "   host:port - change default 127.0.0.1:12344 to another host and/or port"
            exit 0
            ;;
        *)
            if [ -z "$gpus" ]; then
                gpus="$1"
            elif [ -z "$hostport" ]; then
                hostport="$1"
            else
                echo "Too many arguments. At most 2 arguments are accepted."
                exit 1
            fi
            ;;
    esac
    shift
done

if [ -n "$gpus" ]; then
    export CUDA_VISIBLE_DEVICES="$gpus"
else
    echo "Warning: No devices provided. Using all GPUS."
fi

if [ -n "$hostport" ]; then
    export OLLAMA_HOST="$hostport"
fi

echo "Run your client as"
echo
echo "    OLLAMA_HOST=$hostport $OLLAMA_BIN list"
echo

# Start Ollama server in background
$COMMAND &

# Wait a moment for server to come up
sleep 2

# Automatically run gemma3:4b
echo "Starting model: gemma3:4b"
$OLLAMA_BIN run gemma3:4b
