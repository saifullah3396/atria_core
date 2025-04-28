## Environment Setup

### Create a virtual environment
Install poetry:
```
sudo apt install python3-poetry
```

```
pip install --user virtualenv virtualenvwrapper
echo export WORKON_HOME=$HOME/.virtualenvs >> ~/.bashrc
echo source ~/.local/bin/virtualenvwrapper.sh >> ~/.bashrc
source ~/.bashrc
```

```
mkvirtualenv atria
workon atria
```

### Install from git
Install the build dependencies:
```
poetry install git+https://git.opendfki.de/saifullah/atria.git@1.0.0
```
### Install from source
Install the dependencies:
```
poetry lock
poetry install
```

Build atria hydra configurations:
```
python -m atria._hydra.build_configurations
```

Setup environment variables:
```
export PYTHONPATH=<path/to/atria>/src
```
