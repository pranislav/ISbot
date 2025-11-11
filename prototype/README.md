# Prototype

**Hey, You, dear developer, don't even try to get help from LLMs to do the setup!**
**Read the docs FFS, it is easier and IT WORKS!**

**[Install OpenWebUI](https://docs.openwebui.com/getting-started/quick-start):**
```sh
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install --verbose open-webui  # This will take some time.
pip install -r pipelines/requirements.txt  # This will take even more time.

# There are some compatibility issues we cannot fix.
pip check
```

**[Run OpenWebUI](https://docs.openwebui.com/getting-started/quick-start):**
```sh
source venv/bin/activate
open-webui serve
```

**Forward OpenWebUI to local device:**
```sh
ssh -fN -L 8080:localhost:8080 apollo`
```

**Start Pipeline server (pipelines stored at `pipelines/pipelines/`):**
```sh
cd pipelines  # Go to pipelines project dir (the one containing requirements.txt)
./start.sh
```

Run Ollama (see Workarounds section).

See [tutorial](https://docs.openwebui.com/getting-started/quick-start/starting-with-ollama) to connect OpenWebUI to Ollama models - add local connection without auth.

Also add pipelines server (see [tutorial](https://open-webui.com/pipelines/)):
 1. Add connection type to `Local`, URL to `http://localhost:9099` and bearer API token to `0p3n-w3bu!`.
 1. Ignore any errors that are showing on server side regarding spec, it should not be a problem hopefully.


## Workarounds

I have no permissions to run Ollama on Apollo, therefore I forward it from Aura:
```sh
ssh -fN -L 11434:localhost:11434 aura
```

Test the Ollama conection:
```sh
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:14b",
  "prompt":"Why is the sky blue?"
}'
```

Prefered solution is to add Podman to servers (rootless docker) that would allow us to
install Ollama images. Other solution is to install Ollama locally, but this would not
allow us to update it.
