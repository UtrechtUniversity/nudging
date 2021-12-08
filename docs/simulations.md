
### Simulations
We generated 1000 datasets with varying control parameters such as noise, number of subjects, etc. as described [here](simulations_supplementary.md). Each dataset consists of half treatment group (nudged) and half control group (non-nudged). For each dataset we determine the nudge effectiveness as described above. We then compute the correlation between nudge effectiveness and the CATE. If the correlation is good, we may assume that the nudge effectivenes is a good proxy for the CATE. Note that our goal is not neccesarily to predict the CATE but rather to find a proxy of the CATE that we can use on combined studies/datasets with different outcome variables.

For the simulated datasets, we know the CATE that was used as model input, we call this *cate_model*. We can also derive the "observed" CATE (*cate_obs*) per subgroup by mean(treatment) - mean(control). While *cate_model* is known both for individuals and subgroups, *cate_obs* can only be derived for subgroups. We show the correlation for all three types of CATEs.

Except for the method as described above (probabilistic model), we also apply a machine learning model that directly predicts the outcome and CATE, as described [here](models_supplementary.md). We refer to the latter as the CATE regression model, since regression is used to fit the CATE. We expect the CATE regression model to be more accurate in predicting the outcome/CATE within one dataset, in particular when there are many covariates. However, it has the limitation that it can only be applied to combined studies if the outcome variables are standardized. How to standardize the data for this use is a topic for future investigation. In the meantime, we can use the CATE regression model to compare with the probabilistic model.

For the below plots, CATE regression model=0 (blue) and probabilistic model=1 (orange).

#### Correlation with *cate_obs* for subgroups:
![subgroups_cate_obs](../plots_subgroups_cate_obs/noise_frac.png)

#### Correlation with *cate_model* for subgroups:
![subgroups_cate_model](../plots_subgroups_cate_model/noise_frac.png)

#### Correlation with *cate_model* for individuals:
![ind_cate_obs](../plots_ind_cate_model/noise_frac.png)
