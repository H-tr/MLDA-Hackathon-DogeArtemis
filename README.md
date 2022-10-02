# MLDA-Hackathon-DogeArtemis
We used Python 3.9.9 and PyTorch 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.7 or later and recent PyTorch versions. The codebase also depends on a few Python packages, most notably HuggingFace Transformers for their fast tokenizer implementation and ffmpeg-python for reading audio files. The following command will pull and install the latest commit from this repository, along with its Python dependencies

```pip install git+https://github.com/openai/whisper.git ```
It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

# on Ubuntu or Debian
```sudo apt update && sudo apt install ffmpeg```

# on Arch Linux
```sudo pacman -S ffmpeg```

# [on MacOS using Homebrew](https://brew.sh/)
```
brew install ffmpeg
```

# [on Windows using Chocolatey](https://chocolatey.org/)

```choco install ffmpeg```

# [on Windows using Scoop](https://scoop.sh/)

scoop install ffmpeg
You may need rust installed as well, in case tokenizers does not provide a pre-built wheel for your platform. If you see installation errors during the pip install command above, please follow the Getting started page to install Rust development environment. Additionally, you may need to configure the PATH environment variable, e.g. export PATH="$HOME/.cargo/bin:$PATH". If the installation fails with No module named 'setuptools_rust', you need to install setuptools_rust, e.g. by running:
```
pip install setuptools-rust
```
