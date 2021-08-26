# Precision nudging with Machine Learning

## Introduction
This is the code for the precision nudging project. The scientific aim of this project is to develop predictive models with Machine Learning in order to determine the most effective nudge for persons, given the nudging goal and the individual personal circumstances. Most nudging research uses standard social science techniques like field experiments, surveys, or document analyses. Using Machine Learning helps us discover new ways to apply behavior change techniques to solve societal problems. We focus on improving health behavior, such as eating and exercising, as unhealthy behavior is a crucial societal problem. There is data available (from open data of nudging studies) that we can use to train our model. 

Our method consists of two steps: first, we combine open data from different, published studies, and then we use a classifier to determine the most effective nudge per subject group. In order to combine the heterogenous data, we calculate per subject the nudge success using propensity score matching. The propensity score is the probability of treatment assignment, given observed baseline characteristics. In our case, the treatment group is the group that received a nudge. The observed baseline characterics are currently the age and gender of the subject, although these could easily be expanded depending on included studies. In the sections below, we elaborate on these two steps.

## Combining data from different studies
One of the main challenges is how to combine the data from the widely varying studies. Each study has measured a different outcome variable to determine the effectiveness of a nudge. Furthermore, typically the effectiveness of a nudge is determined through an observational (non-randomized) study and not a randomized controlled trial. In an observational study, the treatment and control (untreated) groups are not directly comparable, because they may systematically differ at baseline. Here, we propose to use propensity score matching to tackle these issue (see e.g. [Austin 2011](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)). The propensity score can be used to balance the treatment and control groups to make them comparable. [Rosenbaum and Rubin (1983)](https://academic.oup.com/biomet/article/70/1/41/240879) showed that treated and untreated subjects with the same propensity scores have identical distributions for all baseline variables. Thus, the propensity score allows one to analyze an observational study so that it mimics a randomized controlled trial.

For each study separately, we estimate the propensity score by logistic regression. This is a statistical model used to predict the probability that an event occurs. In logistic regression, the dependent variable is binary; in our case, we have Z=1 for the treated subjects and Z=0 for the untreated subjects. We can then derive the logistic regression model and subsequently use it to calculate the propensity score for each subject. Propensity score matching is done by nearest neighbour matching of treated and untreated subjects, such that matched subjects have similar values of the propensity score.

When we have matched subjects, we simply determine the nudge succes by evaluating whether the outcome variable had increases or decreased, depending on the nudge study. Thus nudge success is a binary, 0 for failure or 1 for success, which allows us to combine the results for different studies.

Finally, we record for each subject in the treatment group the following:
- age (in decades)
- gender (0=female, 1=male)
- nudge succes (0=failure, 1=success)
- nudge domain
- nudge type

The latter two, nudge domain and nudge type, can differ per study. We distinguish the following categories in this study:

Nudge types (see [Sunstein (2014)](https://link.springer.com/article/10.1007/s10603-014-9273-1)):
1. Default. For instance automatic enrollment in programs 
2. Simplification 
3. Social norms 
4. Change effort 
5. Disclosure 
6. Warnings, graphics. Think of cigarettes. 
7. Precommitment 
8. Reminders 
9. Eliciting implementation intentions 
10. Feedback: Informing people of the nature and consequences of their own past choices 

Nudge domains (see [Hummel and Maedche (2019)](https://ideas.repec.org/a/eee/soceco/v80y2019icp47-58.html)):
1. Energy consumption 
2. Healthy products 
3. Environmentally friendly products 
4. Amount donated 


## Probabilistic classifier for precision nudging
Once, we have combined the data from different studies, we can determine which nudge type is most effective for a certain group of people, for a given nudge domain. In this study, we use age and gender to divide people into subgroups, although as said before we could easily include more observed characteristics if these are available. We use a probabilistic classifier to determine the most effective nudge, which has the advantage that we can also rank nudges on effectiveness instead of selecting only the most effective one. We implemented both a logistic regression and a naive Bayes classifier using [scikit-learn] (https://scikit-learn.org). Logistic regression is a discriminitive model, meaning it learns the posterior probability directly from the traning data. Naive Bayes is a generative model, meaning it learns the joint probability distribution and uses Bayes Theorem to predicts the posterior probability. Typically, naive Bayes converges quicker but has a higher error than logistic regression, see [Ng and Jordan 2001] (https://dl.acm.org/doi/10.5555/2980539.2980648). Thus, while on small datasets naive Bayes may be preferable, logistic regression is likely to achieve better results as the training set size grows.



## Installation
This project makes use of [Poetry](https://python-poetry.org/) for managing dependencies. You can simply install the dependencies with: 

`poetry install`

Note that the `poetry run` command executes the given command inside the project’s virtualenv.

## Run software

### Get data
The data used in this project is under DVC version control. To get access to the data contact one of the repo contributors. The following assumes that the external data has been downloaded.

### Calculate nudge succes per subject
`poetry run python src/nudge_success.py`

### Calculate probability of nudge succes per subgroup  
`poetry run python src/nudge_probability.py`

By default logistic regression is used, but you can also select naive Bayes.
