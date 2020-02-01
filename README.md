# sample machine learning for binary text classification

## Objective

This sample demonstrate how to use ML to classify user humor in like (1) or dislike (0) 
using supervised learning.

This is a binary classification but you can extend to multiples categories using some data techniques 
and changing the model from `binary_crossentropy` to `categorical_crossentropy`


## What you need

- python 2.7 ~ 3.x (used v 3.7.6)
- keras
- tensorflow OR theano OR cntk (used tensorflow)
- some python libs (see bellow)


### install tensorflow

```

# Current stable release for CPU and GPU
% pip3 install tensorflow --user

```

### install python libs
```
pip3 install pandas 
pip3 install sklearn
pip3 install matplotlib
```

### Run

```
% python3 classify.py
```

### Result

```
Using TensorFlow backend.
Training Accuracy: 0.9973
Testing Accuracy:  0.8400
```

![result](https://github.com/rharari/text-classification/blob/master/result_plot.png)
