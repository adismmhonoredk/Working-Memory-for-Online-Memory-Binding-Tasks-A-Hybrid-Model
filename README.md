# Working Memory for Online Simple Bindings: A Hybrid Model

Created by Mahdi Heidarpoor on 8/5/20.

Copyright © 2020 Mahdi Heidarpoor. All rights reserved.
__________________________________________________________________________________________________________
## TABLE OF CONTENTS
__________________________________________________________________________________________________________
## Codes
__________________________________________________________________________________________________________
* Dependencies:
```
bash
dill
logging
matplotlib.pyplot
numpy
pickle
tensorflow>=2
```

* In all Jupyter files we create train and validation (sometimes its name is 'test') data and save the sets, then create the model and save it. Or if you want to load previous data and model, you must outline saving new data and model and load previous ones. 

* Notice: In some tasks, we take validation test at end of each epoch. We separate the file of validation test from other codes in 'Second-order task' to simplify codes, but, in 'a cue-based binding task' we take validation test in the same file.

* We break our train phase by loops so we can save the model between our epochs. set loops number to 0 if you want to skip the train section.

* At the end of the Jupyter filse, we create  test_data just to show you our model performance and IV changes.

* All requested codes of paper in order to appear in the paper.

### 2 Method

##### 2-1 Model's python code (Each Task have a separate model in its folder exactly the same as this model)
##### 2-1-2 Testing Balanced Random network in Jupyter
##### 2-2 Trainer's python code (Each Task have a separate trainer in its folder the same as this model)
##### 2-3 Testing other Networks in tasks (We recommend to see results code at first)
##### 2-3-1 Direct connection to layer three
##### 2-3-2 Ballanced but nonrandom

### 3 Results

##### 3-1 First-Order Memory Binding Task
##### 3-2 Generalized First-Order Binding Task
##### 3-3 Second-order Memory Binding Task
##### 3-4 A Cue-Based Memory Binding Task
```You must Extract "trained_dcwm_pickle.7z" in the folder to load the trained model without error.```

### 4 Concolusion

##### Third-order  memory binding task: Validation tests
##### Forth-order memory binding task: Validation tests
__________________________________________________________________________________________________________

## HQ Figs
__________________________________________________________________________________________________________
* High-quality figures of paper in order appear in the paper.

* We highly recommend you to see paper HQ figures.

__________________________________________________________________________________________________________
