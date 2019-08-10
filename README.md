# Autocomplete Service

**Using character level language models for autocomplete applications**

## Setup
**Prerequisite Python 3.6**
* Clone the repository `git clone git@github.ibm.com:rgoyal04/autocomplete.git`
* To install dependencies perform `pip install -r requirements.txt`

## Data Preprocessing
Steps for preprocessing new data, to be used for training:
1. Create a text file with each query in a new line. For instance, data for training a model that autocompletes country names will look as follows, with each country name on a new line:
```
India
Sri Lanka
Brazil
United States of America
Australia
...
...
Pakistan
Mexico
China
```
2. Call `python preprocess.py --raw /path/to/raw_data --processed /path/to/processed_data`
    * `/path/to/raw_data` is the path to text file containing raw data created in step 1
    * `/path/to/processed_data` is path of file where processed data will be saved
3. A processed data file will be created at `/path/to/processed_data`. Make sure that directory in which file is to be created exists. This file is not a plain text.

## Training Steps
Steps to train a model on the processed data:
1. Call `python main.py --data /path/to/processed_data --save /path/to/trained_model` to train model
2. Though default values for all hyperparameters have been set, call `python main.py --help` to print out all hyperparameter related options, and other flags available
3. If a machine with GPU is available call `python main.py --cuda --data /path/to/processed_data --save path/to/trained_model` for faster training

## Interactive Demo
A command line demonstraction for viewing results:
1. Call `python demo.py --data /path/to/processed_data --checkpoint /path/to/trained_model`
2. A trained model file `models/subject_title.pt` and corresponding data file `data/chardict.pt` is included and can be used to run the demo as follows `python demo.py --data data/chardict.pt --checkpoint models/subject_title.pt`
3. Call `python demo.py --help` for more demo related options and help

## Flask Demo
Host a flask based autocomplete web service:
1. Call `python flask_demo.py --cuda --data /path/to/processed_data --checkpoint /path/to/trained_model`
2. Use the `--cuda` flag only if a GPU is available for running the code.
3. Example curl command for testing the service:

    `curl -X POST XXX.XXX.XXX.XXX:4001/demo -H 'content-type: application/json' -d '{"prompt": "xyz"}'`

    Where `XXX.XXX.XXX.XXX` should be replaced by IP address of machine where service is hosted and `xyz` should be replaced by the actual query.
