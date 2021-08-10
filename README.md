# nudging-code
This is the code for the precision nudging project. The scientific aim of this project is to develop predictive models for nudges in order to determine the most successful nudge for persons, given the nudging goal and the individual personal circumstances. We will use Machine Learning to develop such nudges. Machine Learning techniques have not been applied to nudging research. Most research uses standard social science techniques like field experiments, surveys, or document analyses. Using Machine Learning helps us discover new ways to apply behavior change techniques to solve societal problems. We focus on improving health behavior, such as eating and exercising, as unhealthy behavior is a crucial societal problem. There is data available (from open data of nudging studies) that we can use to train our model. 

Our approach consists of two steps: we first combine data from different studies, and then we use a classifier to compute the most successfull nudge per subject group. In order to combine the data, we calculate per subject the nudge success using propensity score matching. The propensity score is the probability of treatment assignment conditional on observed baseline characteristics. In our case, the treatment group is the group that received a nudge. The observed baseline characterics are currently the age and gender of the subject, although depending on included studies, these can be expanded. In the sections below, we elaborate on these steps

## Combining data from different studies
One of the main challenges is how to combine the data from the widely varying studies. Each study may have measured a different outcome variable to determine the effectiveness of a nudge. Furthermore, typically the effectiveness of a nudge is determined through an observational study and not a randomized controlled trial. Here, we propose to use propensity score matching to tackle these issue. For an introduction on propensity score, see e.g. Austin 2011 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/).

The propensity score is the probability of treatment assignment given observed baseline characteristics (such as age and gender). In an observational study, the treated and untreated groups are not directly comparable, because they may systematically differ at baseline. The propensity score can be used to balance the study groups to make them comparable. Rosenbaum and Rubin (1983) showed that treated and untreated subjects with the same propensity scores have identical distributions for all baseline variables. Thus, the propensity score allows one to analyze an observational (nonrandomized) study so that it mimics a randomized controlled trial.

For each study separately, we estimate the propensity score by logistic regression. This is a statistical model used to predict the probability that an event occurs. In logistic regression, the dependent variable is binary; Z=1 is the value for the treatment and the value for the control is Z=0. We determine the logistic regression model and subsequently calculate the propensity score for each subject. Propensity score matching is then done by nearest neighbour matching of treated and untreated subjects, such that matched subjects have similar values of the propensity score.

When we have matched subjects, we can simply determine nudge succes by evaluating whether the outcome variable had increases or decreased, depending on the nudge study. Thus nudge success is a binary, 0 for failure or 1 for success.

Finally, we record for each subject in the treatment group the following:
- age
- gender
- nudge succes 
- nudge domain
- nudge success
The latter two, nudge domain and nudge type, can differ per study.

## Classifier for precision nudging





## Installation
This project makes use of [Poetry](https://python-poetry.org/) for managing dependencies. You can simply install the dependencies with: 

`poetry install`

Note that the `poetry run` command executes the given command inside the project’s virtualenv.

## Run software

### Get data
The data used in this project is under DVC version control. To get access to the data contact one of the repo contributors. The following assumes that the external data has been downloaded.

### Calculate nudge succes per participant
`poetry run python src/nudge_success.py`

### Run the Naive Bayes Classifier  
`poetry run python src/classifier_nb.py`
