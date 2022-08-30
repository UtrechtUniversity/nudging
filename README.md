[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UtrechtUniversity/nudging/HEAD?labpath=examples%2Ftutorial.ipynb)

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

Thus, we are interested in the heterogeneous treatment effect of nudges. Conventionally, heterogeneous treatment effects are found by dividing the study data into subgroups (e.g., men and women, or by age) and comparing the conditional average treatment effect (CATE) between subgroups. A challenge with this approach is that each subgroup may have substantially less data than the study as a whole, and there may not be enough data to accurately estimate the effects on subgroups. Recently and increasingly, however, machine learning is used to estimate heterogeneous treatment effects, even to the level of individuals, see e.g. [Künzel et al. 2019](https://www.pnas.org/content/116/10/4156). In this study, we apply different machine learning models to determine the heterogenity of treatment effects.

To determine which nudge is most effective for a certain subgroup of individuals, one could set up a large study where different nudges are applied. However, this would be very time-consuming and costly. Here, we explore the possibiltiy of using open data from previously published nudging studies to train our models. If we are able to use and combine existing datasets, we can study the heterogeneous effects of nudges on a large scale with limited effort.

The project can be split in several steps:
- [We investigate different methods of determining the heterogeneity of treatment effects](docs/methods.md);
- [We create realistic synthetic data to compare the performance of the different methods](docs/simulations.md);
- We investigate the validity of the methods on open data from published studies.


## Online Tutorial

We have a quick online [tutorial](https://mybinder.org/v2/gh/UtrechtUniversity/nudging/HEAD?labpath=examples%2Ftutorial.ipynb).

## Getting Started

### Prerequisites

Install a version of Python>=3.6 for this project.

### Installation (locally)

Open a command line terminal (not a python interpreter!).

Clone the repository with `git clone https://github.com/UtrechtUniversity/nudging.git`.

Go into the newly created directory `cd nudging` (Linux/MacOS).

Install the package with `pip install .`

### Tutorial

The tutorial is available under `examples/tutorial.ipynb`.


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
