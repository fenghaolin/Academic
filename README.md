# Github Page (Academic) of H. Feng
## Introductory materials and tutorials

### Setup your toolbox
* For data science and machine learning, [R](https://www.r-project.org/) and [Python](https://www.python.org) are two mostly used programming language. For scientific computing, another two popular choices are [MATLAB](https://www.mathworks.com/products/matlab.html) and [Julia](https://julialang.org/). I teach programming related courses with MATLAB, R, and Python. 
  * MATLAB is expensive to use for commercial purpose, but many universities have subscribed to some of their academic licenses. In addition, the [GNU](https://www.gnu.org/) community offers the excellent open source [Octave](https://www.gnu.org/software/octave/index), which shares the same language syntax with MATLAB. The advantage of using MATLAB is that, it is extremely easy to learn -- you can probably start using it to write your own research code after a few hours of reading online tutorials.
  * R, Python, and the relative new Julia are all open source, and each of them has its own strength. R is famous for statistics and data visualization, while Python is a general-purpose programming language wildly used in many areas. On the other hand, Julia offers excellent speed.
* A popular development environment is Anaconda, which can bundle Python, R, and Julia together. [Here](./MachineLearning/About_Conda.html) is a very simple tutorial (in Chinese) I provided for my students regarding how to set up Anaconda.
### Machine Learning related
+ [Here](./MachineLearning/Script_Demo_GradientDescent_ADADELTA.html) are some tests regarding introductory example comparing the following algorithms. (The tests are mainly for educational purpose.)
  + Gradient Descent (whose wiki page can be found [here](https://en.wikipedia.org/wiki/Gradient_descent)). An illustrative implementation is given [here](MachineLearning/ML_GradientDescent.m)
  + [ADADELTA](https://arxiv.org/pdf/1212.5701.pdf) by Matthew Zeiler. An illustrative implementation is given [here](MachineLearning/adadelta.m)
+ Machine Learning can be used to solve Dynamic Programming (DP) problems approximately. DP is a powerful and widely used tool in operations research. For many problems, DP is a viable tool. For example the shortest path problem (see [here](./SDP/SPP.html) for the code and example I provide). But more often, its computation complexity is forbidding, mostly due to the famous *curse-of-dimensionality*. [This site](https://castlelab.princeton.edu/) has a lot of useful information regarding Approximate Dynamic Programming (ADP) and learning.
+ Regarding using ADP to overcome DP's curse of dimensionality, [here](https://doi.org/10.1016/j.tre.2021.102508) is a recent paper I co-authored on a problem of eCommerce oriented automated warehouse optimization. We use 'rollout', a technique from reinforcement learning, to overcome the difficulty inherited from the dynamic nature of the problem. The paper was published in Transportation Resarch Part E.

### Statistics/Analytics
#### Conditional Probability and binary classification
+ The simple form of the famous [Bayes rule](https://en.wikipedia.org/wiki/Bayes%27_theorem) is the following: `P(A and B)=P(A|B)*P(B)=P(B|A)*P(A)`.
+ It is very useful in understanding a lot of important concepts and techniques, and here we briefly explain one of them--binary classification.
+ [Here](./Statistics_Analytics/COVID_sensitivity_specificity.nb.html) is an example of conditional probability illustrating some related concepts in binary classification. It is about the COVID-19 fast testing using some real data together with some assumptions.