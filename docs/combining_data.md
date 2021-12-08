### Combining Data

One of the main challenges is how to combine data from the widely varying studies. Each study has measured a different outcome variable to determine the effectiveness of a nudge. Furthermore, in some studies, the effectiveness of a nudge is determined through an observational (non-randomized) study and not a randomized controlled trial. In an observational study, the treatment and control (untreated) groups are not directly comparable, because they may systematically differ at baseline. Here, we propose to use propensity score matching to tackle these issue (see e.g. [Austin 2011](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)).

The propensity score is the probability of treatment assignment, given observed baseline characteristics. The propensity score can be used to balance the treatment and control groups to make them comparable. [Rosenbaum and Rubin (1983)](https://academic.oup.com/biomet/article/70/1/41/240879) showed that treated and untreated subjects with the same propensity scores have identical distributions for all baseline variables. Thus, the propensity score allows one to analyze an observational study as if it were a randomized controlled trial. In our case, the treatment group is the group that received a nudge and the control is the group that didn't. The observed baseline characteristics are specified per study, and typically include age and gender of the subject.

For each study separately, we estimate the propensity score by logistic regression. This is a statistical model used to predict the probability that an event occurs. In logistic regression, the dependent variable is binary; in our case, we have Z=1 for the treated subjects and Z=0 for the untreated subjects. We can then derive the logistic regression model and subsequently use it to calculate the propensity score for each subject. Propensity score matching is done by nearest neighbour matching of treated and untreated subjects, so that matched subjects have similar values of the propensity score.

When we have matched subjects, we simply determine the nudge succes by evaluating whether the outcome variable had increased or decreased, depending on the nudge study. Thus nudge success is a binary, 0 for failure or 1 for success, which allows us to combine the results for different studies.

Finally, we record for each subject in the treatment group the following:
- age
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
