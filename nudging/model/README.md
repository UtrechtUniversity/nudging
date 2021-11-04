## Machine Learning Models

We distinguish between probabilitic models that attempt to predict nudge effectiveness and regressor model that aim to predict the outcome variable and cate.

There are two different kinds of implemented probablistic model implemented: logistic regression and naive bayes. Logistic regression is a discriminitive model, meaning it learns the posterior probability directly from the tranining data. Naive Bayes is a generative model, meaning it learns the joint probability distribution and uses Bayes' Theorem to predicts the posterior probability. Typically, naive Bayes converges quicker but has a higher error than logistic regression, see [Ng and Jordan 2001](https://dl.acm.org/doi/10.5555/2980539.2980648). Thus, while on small datasets naive Bayes may be preferable, logistic regression is likely to achieve better results as the training set size grows.

There are three different kinds of implemented regressor models: S-learners, T-learners and X-learners, see this [paper](https://doi.org/10.1073/pnas.1804597116). S-learners simply apply the regression model directly to the dataset. T-learners apply a regression model to both the nudging and control groups. X-learners are similar to T-learners, but add another model that crosses over the results of the nuding and control groups.

All models are based on [scikit-learn](https://scikit-learn.org).