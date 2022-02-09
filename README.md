# Github Page (Academic) of H. Feng
## Introductory materials and tutorials
### Machine Learning related
+ [Here](./MachineLearning/Script_Demo_GradientDescent_ADADELTA.html) are some tests regarding introductory example comparing the following algorithms. (The tests are mainly for educational purpose.)
  + Gradient Descent (whose wiki page can be found [here](https://en.wikipedia.org/wiki/Gradient_descent)). An illustrative implementation is given [here](MachineLearning/ML_GradientDescent.m)
  + [ADADELTA](https://arxiv.org/pdf/1212.5701.pdf) by Matthew Zeiler. An illustrative implementation is given [here](MachineLearning/adadelta.m)
+ Machine Learning can be used to solve Dynamic Programming (DP) problems approximately. DP is a powerful and widely used tool in operations research, but its computation complexity is sometimes forbidding, mostly due to the famous *curse-of-dimensionality*. [This site](https://castlelab.princeton.edu/) has a lot of useful information regarding Approximate Dynamic Programming (ADP) and learning.
+ Regarding using ADP to overcome DP's curse of dimensionality, [here](https://doi.org/10.1016/j.tre.2021.102508) is a recent paper I co-authored on a problem of eCommerce oriented automated warehouse optimization. We use 'rollout', a technique from reinforcement learning, to overcome the difficulty inheritated from the dynamic nature of the problem. The paper was published in Transportation Resarch Part E.

### Statistics/Analytics
#### Conditional Probability and binary classification
+ The simple form of the famous [Bayes rule](https://en.wikipedia.org/wiki/Bayes%27_theorem) is the following: P(A and B)=P(A|B)*P(B)=P(B|A)*P(A).
+ It is very useful in understanding a lot of important concepts and techniques, and here we briefly explain one of them--binary classification.
+ [Here](./Statistics_Analytics/COVID_sensitivity_specificity.nb.html) is an example of conditional probability illustrating some related concepts in binary classification. It is about the COVID-19 fast testing using some real data together with some assumptions.