# Installation instructions for Ubuntu

## Set up python env
The following was created using <a href="https://www.tensorflow.org/install/pip?lang=python3">TF "Install TensorFlow with pip" page</a>.

1. Install Python related:
```
sudo apt update
sudo apt install python3-dev python3-pip python3-tk
sudo pip3 install -U virtualenv  # system-wide install
``` 
2. Create a virtual environment
```
virtualenv -p python3 ~/venv/mesh_walker
source ~/venv/mesh_walker/bin/activate
pip install --upgrade pip
cd <mesh-walker-path>
pip install -r requirements.txt
```

## More common system installations
```
sudo apt install doublecmd-common
sudo apt install meshlab
sudo snap install pycharm-community --classic
sudo apt install graphviz
```
