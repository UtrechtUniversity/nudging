# Precision nudging with Machine Learning


<!-- TABLE OF CONTENTS -->
## Table of Contents
- [About the Project](#about-the-project)
   - [Combining Data](#combining-data)
   - [Probabilistic Classifier](#probabilistic-classifier)
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
This is the code for the Precision Nudging project. The scientific aim of this project is to develop predictive models with Machine Learning in order to determine the most effective nudge for persons, given the nudging goal and the individual personal circumstances. Most nudging research uses standard social science techniques like field experiments, surveys, or document analyses. Using Machine Learning helps us discover new ways to apply behavior change techniques to solve societal problems. We focus on improving health behavior, such as eating and exercising, as unhealthy behavior is a crucial societal problem. We use open data from nudging studies to train our model. 

Our method consists of two steps: 
1) We combine open data from different, published studies. 
2) We use a classifier to determine the most effective nudge per subject group.

In the sections below, we elaborate on these two steps.

### Combining Data
One of the main challenges is how to combine the data from the widely varying studies. Each study has measured a different outcome variable to determine the effectiveness of a nudge. Furthermore, typically the effectiveness of a nudge is determined through an observational (non-randomized) study and not a randomized controlled trial. In an observational study, the treatment and control (untreated) groups are not directly comparable, because they may systematically differ at baseline. Here, we propose to use propensity score matching to tackle these issue (see e.g. [Austin 2011](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)).

The propensity score is the probability of treatment assignment, given observed baseline characteristics. The propensity score can be used to balance the treatment and control groups to make them comparable. [Rosenbaum and Rubin (1983)](https://academic.oup.com/biomet/article/70/1/41/240879) showed that treated and untreated subjects with the same propensity scores have identical distributions for all baseline variables. Thus, the propensity score allows one to analyze an observational study as if it were a randomized controlled trial. In our case, the treatment group is the group that received a nudge and the control is the group that didn't. The observed baseline characteristics are specified per study, and typically include age and gender of the subject.

For each study separately, we estimate the propensity score by logistic regression. This is a statistical model used to predict the probability that an event occurs. In logistic regression, the dependent variable is binary; in our case, we have Z=1 for the treated subjects and Z=0 for the untreated subjects. We can then derive the logistic regression model and subsequently use it to calculate the propensity score for each subject. Propensity score matching is done by nearest neighbour matching of treated and untreated subjects, so that matched subjects have similar values of the propensity score.

When we have matched subjects, we simply determine the nudge succes by evaluating whether the outcome variable had increased or decreased, depending on the nudge study. Thus nudge success is a binary, 0 for failure or 1 for success, which allows us to combine the results for different studies.

Finally, we record for each subject in the treatment group the following:
- age (in decades)
- gender (0=female, 1=male)
- other relevant personal characteristics
- nudge success (0=failure, 1=success)
- nudge domain
- nudge type

Note that the nudge domain and nudge type can differ per study. We distinguish the following categories in this study:

**Nudge types** (see [Sunstein (2014)](https://link.springer.com/article/10.1007/s10603-014-9273-1)):
1. Default. For instance automatic enrolment in programs 
2. Simplification 
3. Social norms 
4. Change effort 
5. Disclosure 
6. Warnings, graphics. Think of cigarettes. 
7. Precommitment 
8. Reminders 
9. Eliciting implementation intentions 
10. Feedback: Informing people of the nature and consequences of their own past choices 

**Nudge domains** (see [Hummel and Maedche (2019)](https://ideas.repec.org/a/eee/soceco/v80y2019icp47-58.html)):
1. Energy consumption 
2. Healthy products 
3. Environmentally friendly products 
4. Amount donated 


### Probabilistic Classifier
Once, we have combined the data from different studies, we can determine which nudge type is most effective for a certain group of people, for a given nudge domain. We use age and gender to divide people into subgroups, although we could easily include more observed characteristics if these are available. We use a probabilistic classifier to determine the most effective nudge, which has the advantage that we can also rank nudges on effectiveness instead of selecting only the most effective one. Nudge effectiveness is defined as the probability of nudge success.

We implemented both a logistic regression and a naive Bayes classifier using [scikit-learn](https://scikit-learn.org). Logistic regression is a discriminitive model, meaning it learns the posterior probability directly from the tranining data. Naive Bayes is a generative model, meaning it learns the joint probability distribution and uses Bayes' Theorem to predicts the posterior probability. Typically, naive Bayes converges quicker but has a higher error than logistic regression, see [Ng and Jordan 2001](https://dl.acm.org/doi/10.5555/2980539.2980648). Thus, while on small datasets naive Bayes may be preferable, logistic regression is likely to achieve better results as the training set size grows.

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
