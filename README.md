# ML-Project
Our goal is to build a sequence labelling system using the Hidden Markov Model (HMM) and then use the system to predict part-of-speech (POS) tag sequences for new sentences.

## Getting Started
The python scripts and prediction scores (F scores) are all in the main directory. Prediction score files are labelled with the part they are for. i.e. prediction scores for Part 2 is labelled `p2_results`, and so on. Train and test data are stored in the respective folders labelled by the dataset (EN/FR/CN/SG). 

### Prerequisites
The project is done in Python 3.6.7. Make sure you are running Python 3. 

### Running the files
sharedFunctions.py contains all the functions shared across Parts 2, 3 and 4. You cannot run this file alone but it is required for the other parts to run. Functions in sharedFunctions.py include `estEmissions()` and `estTransitions()`.

#### Predicting
To perform predictions for a desired Part, just do

```
> python part[2/3/4/5].py
```

A prediction file will be generated for each of the datasets (EN/FR/CN/SG) in their respective folders. For example, running `python part2.py` will generate 4 `dev.p2.out` files, one in each of the EN, FR, CN and SG folders. 

Part 5 is a little different since we need to run it on dev, test and test2. You should see a prompt when you run part5.py. Just follow the prompts accordingly. Here is an example:

```
> python part5.py
Which language do you wish to use? (EN/FR/CN/SG) 
EN

dev, test or test2? 
test

Output: EN/test.p5.out
Done!
```

#### Evaluating F1 Score
We decided to write a script that runs evalResult.py on all the datasets (EN/FR/CN/SG) at once, for any given Part. Just do

```
python evalAll.py p[2/3/4/5]
```

The output will be saved to a neat file in the main directory as `p[2/3/4/5]_results`. For example, to evaluate the F scores for Part 2, running `python evalAll.py p2` will generate the F1 scores in the file `p2_results`.


## Authors
SUTD ISTD Class of 2019

[Rayson Lim 1002026](https://github.com/nosyarlin)

[Kimberlyn Loh 1002221](https://github.com/kimb3rlyn)

[Angelia Lau 1002417](https://github.com/angelialau)

