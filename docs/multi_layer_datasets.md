## Simulation

The main procedure that we have for simulating datasets, uses a layered approach. Each of these layers represents some property of that particular dataset. For an intuitive sense, imagine the layers being toppings of a pizza.

- Base layer that represents how well nudges work in all circumstances.
- Domain layer that represents how nudges work within a specific nudge domain.
- Type layer that represents how nudges work within a specific nudge type.
- Dataset layer that represents how nudges work for a specific dataset/study.

### Correlation matrices

As a basis for generating correlated features and results, we use a method from the [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.random_correlation.html) package. It generates a random correlation matrix with a given set of eigen values. For these eigen values we generate `n_features+2` uniformly distributed between 0 and 1. To allow for different correlation strengths, we take these values to the power `eigen_power`. Finally the eigenvalues are normalized to 1 so that they can generate a valid correlation matrix.

One issue that arises with these correlation matrices is that there are sometimes occasions when one wants to have more control over the values of the correlation matrix. In particular, we might want some features not to be correlated. For example, we do not want gender and age to be correlated. To this effect we have used correlation matrix stretching as described in [this paper](https://epubs.siam.org/doi/abs/10.1137/140996112). With this we can set the number of features that are uncorrelated between them `n_features_uncorrelated` and the number of correlated features `n_features_correlated`. The uncorrelated features are still correlated of course to the outcome and the nudge.


### Making the pizza

To define an ensemble of datasets that are mutually related, we go back to our pizza toppings. To create a correlation matrix we start with dough: the base layer that represents correlations between features for all types of nudges and domains. The thickness of this dough (variance captured by nudging as a principle) is set to one. The thickness of other topping layers is measured relative to the base layer. The thicknesses are defined by variables as follows:

- Base layer: 1
- Domain layer: `nudge_domain_variation` (default: 0.2)
- Type layer: `nudge_type_variation` (default: 0.5)
- Dataset layer: `dataset_variation` (default: 0.5)

Thus, it is assumed with the default values that nudge propensities are for a large part the same among different nudge types/domains and datasets. The domain is assumed to be less of an influence than either the nudge type of the dataset. Naturally all these values can be easily adjusted.

For the domain and type layers a discrete number of different kinds are defined. For the domain layer, there are `n_nudge_domain` number of domains (default: 3), and for the type layer, there are `n_nudge_type` number of types (default: 3). There is of course just one base layer and an infinite number of dataset layers. For each discrete nudge domain and nudge type a correlation matrix is computed using the previously described in the previous [section](#correlation-matrices). To final correlation matrix is a weighted sum of the selected items for the pizza, which amounts by default to the sum of:

- Base layer: 1/2.2 * C_b
- Domain layer: 0.2/2.2 * C_dm
- Type layer: 0.5/2.2 * C_t
- Dataset layer: 0.5/2.2 * C_dt

The result of this is that the correlation matrices of all datasets feature some similarities, and when their domain and/or type are equal they share more similarities.

### Putting the pizza in the oven

In the next stage, the correlation matrix is converted to an actual dataset. The basis of this is [cholesky decomposition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html) of the correlation matrix and multiplying it with a matrix of shape (n_features+2, n_samples) with values that are normally distributed and transposing it. This is the end of the procedures for the features themselves. However, for the outcome and the nudge there is a separate procedure.

For the nudge this is simple: it assigns half the participants as being nudged and the other half as being in the control group.

The outcome is more involved to create. First it is important now to explain why there are two extra features instead of one. The easiest way to define the the outcome would be to just take the extra column in the feature matrix. Thus, the basic idea of having two features determine the outcome is that one more or less determines the response of the control group and the other that of the nudged group. The difference between the two bringing the possibility of targeting certain groups for more effective treatment. Admittedly, this supposes that precision nudging can work. On the other hand, there is a parameter that controls this and in the limit vanishes this possibility.

Now as to the actual procedure. The nudged group simply gets the average treatment effect (`nudge_avg`) added to the nudge column. For the control group, the intermediate outcome is a weighted average of the nudge and control columns of the feature matrix, with the `control_unique` parameter as its weight between 0 and 1, with 1 signifying that the control column is used, and 0 the nudge column. A second parameter that allows for precision nudging is `control_precision`. The control outcome is multiplied by this value that lies between 0 and 1 so that the the outcome is less strongly correlated to the features for the control group than for the nudge group. A larger value than 1 is technically possible and would result in a control group that has stronger differences in outcome.

The final step adds noise, but before noise is added the real conditional average treatment effect (CATE) is computed for each of the participants/samples.The amount of noise is controlled by the parameter `noise_frac`, which adds `noise_frac/(1-noise_frac) * N(0, 1)` amount of noise, with N(0, 1) being the normal distribution with mean 0 and variance 1.

### Generating the parameters

The parameters mentioned in the previous are different for each dataset generated. The determination of the parameters depends on the nudge type and domain. Datasets that have the same nudge type and domain should have similar ranges of parameters (i.e. they are more alike. The procedure is similar to the procedure described [before](#making-the-pizza), and in fact uses the same parameters. The base layer in this case is simply denoted by the base range of the parameters:

- `nudge_avg`: [0, 0.3]
- `noise_frac`: [0.1, 0.9]
- `control_unique`: [0, 1.0]
- `control_precision`: [0.2, 1.0]

Then for each of the layers, we want these intervals to be shrunk proportionally to their weight. The weights are recalculated with the base layer removed:

- Domain layer: 0.2/1.2
- Type layer: 0.5/1.2
- Dataset layer: 0.5/1.2

For example, the shrinking is done with the parameter `nudge_avg` it starts on the interval [0, 0.3], which is first shrunk by 0.2/1.2 \* 0.3 = 0.05 to 0.25. For each domain, a random start of the interval is chosen between 0 and 0.05. Then the same procedure is done for to type layer, which results in an interval size of 0.5/1.2*0.3 for the dataset layer. This procedure ensures that parameters for all datasets are the same, but parameters for datasets with the same domain and/or type are more similar to each other. One side effect of this procedure is that parameters are not uniformly distributed anymore. We don't think this is too much of an issue as long as this is understood to be the result.