# Precision nudging with Machine Learning


<!-- TABLE OF CONTENTS -->
## Table of Contents
- [About the Project](#about-the-project)
   - [Combining Data](docs/combining_data.md)
   - [Probabilistic Model](docs/probabilistic_model.md)
   - [Simulations](docs/simulations.md)
- [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Testing](#testing)
- [Usage](#usage)
   - [Get Data](#get-data)
   - [Prepare](#prepare)
   - [Train](#train)
   - [Evaluate](#evaluate)
   - [Predict](#predict)
- [Contributing](#contributing)
- [Contact](#contact)


## About the Project
This software package is under development within the Precision Nudging project. The scientific aim of this project is to use open data to develop predictive models with Machine Learning, in order to determine the most effective nudge for persons, given the nudging goal and the individual personal circumstances. 

Thus, we are interested in the heterogeneous treatment effect of nudges. Conventionally, heterogeneous treatment effects are found by dividing the study data into subgroups (e.g., men and women, or by age) and comparing the conditional average treatment effect (CATE) between subgroups. A challenge with this approach is that each subgroup may have substantially less data than the study as a whole, and there may not be enough data to accurately estimate the effects on subgroups. Recently and increasingly, however, machine learning is used to estimate heterogeneous treatment effects, even to the level of individuals, see e.g. [Künzel et al. 2019](https://www.pnas.org/content/116/10/4156).

Most nudging research uses standard social science techniques like field experiments, surveys, or document analyses. Using Machine Learning combined with open data can help us discover new ways to apply behavior change techniques to solve societal problems. We focus on improving health behavior, such as eating and exercising, as unhealthy behavior is a crucial societal problem. We use open data from published nudging studies to train our model. 

The project can be split in several steps: 
1) [We combine open data from different, published studies](docs/combing_data.md)
2) [We train a machine learning model on the combined dataset to determine the most effective nudge per subject group](docs/combing_data.md)
3) [We simulate datasets to test the model and compare against other methods](docs/simulations.md)

## Getting Started

### Prerequisites
This project makes use of Python 3.9.2 and [Poetry](https://python-poetry.org/) for managing dependencies. 

### Installation
You can simply install the dependencies with 
`poetry install` in the projects root folder.

### Testing
The test are located in the `tests` folder. To run the tests execute:
`poetry run pytest tests`

Note that the `poetry run` command executes the given command inside the project’s virtual environment.

## Usage
The data processing pipeline consists of several stages which we describe below.

### Get Data
The data used in this project is under [DVC](https://dvc.org/) version control. To get access to the data contact one of the repo contributors. The following assumes that the external data has been downloaded. To check the downloaded data:

`poetry run python nudging/check_data.py`

This should give a summary of the datasets stored in `data/external`.

### Prepare
Calculate nudge succes per subject with:

`poetry run python nudging/prepare.py`

This generates a csv file for each study in `data/interim` and a combined csv file `data/interim/combined.csv`. Each row represents a subject and contains personal characteristics (covariates) and nudge success calculated using propensity score matching. The covariates used for propensity score matching are defined per dataset/study separately.

### Train
First, choose the features to be used for training through the configuration file `config.yaml`. Only datasets containing all features will be used. By default, we use:
- nudge_domain
- nudge_type
- gender
- age

Train a probabilistic classifier on the combined dataset:

`poetry run python nudging/train.py`

By default logistic regression is used, but you can also select naive Bayes with:

`poetry run python nudging/train.py naive_bayes`

The trained model is written to `models/nudging.joblib`.

### Evaluate
Evaluate the trained model using the combined dataset:

`poetry run python nudging/evaluate.py`

 Note that we have not (yet) split off the dataset for training and evaluation. This means we apply the model on the same dataset we trained on. The computed probability is rounded to 0 or 1 and compared to the nudge success. TO DO: evaluate on unused data.

### Predict
Predict nudge effectiveness using the trained model:

`poetry run python nudging/predict.py`

The predicted nudge effectiveness per subgroup is written to `data/processed/nudge_probability.csv`. Also, plots of the nudge effectiveness are generated and stored in the `plots` folder.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact
[UU Research Engineering team](https://github.com/orgs/UtrechtUniversity/teams/research-engineering) - research.engineering@uu.nl
