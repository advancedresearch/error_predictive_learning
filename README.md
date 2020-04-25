# Error Predictive Learning
Black-box learning algorithm using error prediction levels

This is a very simple black-box learning algorithm which
uses higher order error prediction to improve
speed and accuracy of search to find local minima.

See paper about [Error Predictive Learning](https://github.com/advancedresearch/path_semantics/blob/master/papers-wip/error-predictive-learning.pdf)

### Error prediction levels

In error predictive learning, extra terms are added to the error
function such that the search algorithm must learn to predict error,
error in predicted error, and so on.
This information is used in a non-linear way to adapt search behavior,
which in turn affects error prediction etc.

This algorithm is useful for numerical function approximation
of few variables due to high accuracy.

### Reset intervals

In black-box learning, there are no assumptions about the function.
This makes it hard to use domain specific optimizations such as Newton's method.
The learning algorithm need to build up momentum in other ways.

Counter-intuitively, forgetting the momentum from time to time
and rebuilding it might improve the search.
This is possible because re-learning momentum at a local point is relatively cheap.
The learning algorithm can takes advantage of local specific knowledge,
to gain the losses from forgetting the momentum.
