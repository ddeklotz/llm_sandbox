# Models

This directory is for training and building models that will be consumed by the application. I need helpful instructions here because I always seem to bungle up python environments.

## Setup
set up a python virtual environment and install the requisite dependencies.

```
py -m venv env
env\Scripts\activate
pip3 install -r requirements.txt
```

## IDE
Open VS code to the models directory and select the virtual environment's interpreter (bottom right corner)

## Run

make sure the virtual environment is active first.
```
python <script name>
```

## to install a new package

from models directory with virtual environment activated:
```
pip3 install <packagename>
pip3 freeze >requirements.txt
```
