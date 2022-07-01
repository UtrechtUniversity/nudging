## Simulation handbook

This handbook only considers the more simple form of simulation, which generates independent datasets.
For the multi-layered approach, see the more technical [handbook](multi_layer_datasets.md).

### Basics
The simplest and most common way to generate datasets is by using the `generate_datasets` function:

```python
from nudging.simulation import generate_datasets

datasets = generate_datasets(10, avg_correlation=0.1, n_samples=1000)
```

to generate a list of 10 independent datasets, where the average correlation between features is 0.1,
and the number of samples is 1000. There in total 12 parameters that can be set apart from the number
of datasets, which will be described later on. Internally, the dataset generation is done through many
steps in a pipeline. We will discuss these steps in the next few sections.

#### Create correlation matrix (`CorrMatrix`)
First a correlation matrix is generated using the Scipy function [random correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.random_correlation.html). The input for this function is a list of eigenvalues. To create these eigenvalues, we take a series of uniformly distributed values take them to the power "a", where in general higher absolute values of "a" generate correlation matrices with higher correlation. Naturally, this parameter is not very intuitive, and therefore we have designed a method on top of this that converts a desired average correlation with `avg_correlation` to an approximate power "a". With this approximate "a", correlation matrices are generated until one is within a small difference from the target (`avg_correlation`). This matrix is then stretched back to exactly the target value of correlation. Correlations between independent features are not counted. As a note, this average correlation is not necessarily the final correlation is not the same as the correlation of the final feature matrix. Partly this is because there are subsequent transformations that do not preserve the average. Another reason is that the outcome in the correlation matrix is actually represented by two 

There is an option to generate features that are not correlated with each other, which is done by setting the parameter `n_features_uncorrelated`. The number of correlated features is given by `n_features_correlated`.

### Create feature matrix (`CreateFM`)

From a correlation matrix, this pipeline element generates `n_samples` samples of the feature matrix
with the correct correlations between the features. This is done by Chelesky decomposition, and
multiplying it with normally distributed matrix. The shape of this matrix is defined by the number
of wanted/required `n_samples`. The two columns that determine the outcome are now split off and processed.
For each of the samples, a outcome is computed for the two cases of being in the control or treatment group.
For the treatment outcome, it is simply the first of the outcome columns plus the average nudge effect `nudge_avg`.
The control outcome is slightly different. This is inspired by observations of real datasets that models trained
on the treatment group are also pretty good or even better than models trained on the treatment group for
that same treatment groups. The parameter `control_unique` governs how the control group behaves like the
treatment group, and ranges from 0 to 1, where a value of 1 means that its behavior is completely different
while 0 means very similar in terms of correlations. Another observation from real datasets is that while
often the treatment and control groups are modelled by similar models, they do not generally have the same
strength of deviation: treatment groups were found qualitatively to have less dependence of the outcome on
the other features. This introduces another parameter `control_precision`, with which the control outcome
is multiplied after the previous step.


### Introduce non-linearity (`Linearizer`)

This element can transform a linear feature matrix into one that has non-linear dependencies as well. It does so by using a third order polynomial. Each of the columns is transformed as a[0]*x + 0.5*a[1]*x + 0.1*a[2]*x, where `a` is a random vector drawn from a uniform distribution between -0.5 and 0.5, and `x` is the feature vector. For the nudge and control outcomes, `a` is equal, but for all other features, `a` is drawn randomly for each of them. The parameter `linear` determines whether this transformation is applied. If `linear = True`, then the transformation is not applied.

### Add noise (`AddNoise`)

This step simply adds Gaussian noise to the outcomes. The signal to noise ratio is governed by the `noise_frac` parameter, where a `noise_frac` of 0 does not add any noise, while `noise_frac = 1` would be very/infintely noisy (but will generate errors).

### Create matrix data (`CreateMatrixData`)

This step simply converts the feature matrix and outcomes into a dataset that can be trained on by the nudging models.

### Convert feature to age (`ConvertAge`)

This step converts one of the columns to integers between 18 and 80. 

### Convert feature to gender (`ConvertGender`)

Convert one of the columns to a binary variable (integers).

### Create categorical variables (`Categorical`)

Convert `n_rescale` columns to a categorical variable. Each of these new categorical variables have `n_layers` of classes. If `n_layers` is a range, then each of the columns has a number of classes in this range (and they are not necessaryily all equal to each other).
