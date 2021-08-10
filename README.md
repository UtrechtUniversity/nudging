# nudging-code
This is the code for the precision nudging project.

## Installation
This project makes use of [Poetry](https://python-poetry.org/) for managing dependencies. You can simply install the dependencies with: 

`poetry install`

Note that the `poetry run` command executes the given command inside the project’s virtualenv.

## Run software

### get data
The data used in this project is under DVC version control. To get access to the data contact one of the repo contributors.

### Calculate nudge succes per participant
`poetry run python src/get_success.py`

### Run the Naive Bayes Classifier  
`poetry run python src/classifier_nb.py`
