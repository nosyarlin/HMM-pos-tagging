# ML-Project
our goal is to build a sequence labelling system using HMM and then use the system to predict tag sequences for new sentences.

## Getting Started
The python scripts and prediction scores(F scores) are all in the main directory. Prediction score files are labelled with the part they are for. i.e. Prediction scores for part 2 is labelled p2_results, and so on. Train and test data are stored in the respective folders labelled with their language (EN/FR/CN/SG). 

### Prerequisites
The project is done in Python 3.6.7. Make sure you are running Python 3. 

### Running the files
sharedFunctions.py contains all the functions shared across parts 2, 3 and 4. You cannot run this file alone but it is required for the other parts to run. Functions in sharedFunctions.py include estEmissions() and estTransitions()

#### Predicting
To create predictions, just do

```
> python part[2/3/4/5].py
```

The prediction files would be saved in the respective folders labelled with their languages (EN/FR/CN/SG). eg. running part2.py would create 4 files, dev.p2.out for EN, FR, CN and SG. 

Part 5 is a little different since we need to run it on dev, test and test2. You should see a prompt when you run part5.py. Just follow the prompts accordingly. Here is an example.

```
> python part5.py
Which language do you wish to use? (EN/FR/CN/SG) 
EN

dev, test or test2? 
test

Output: EN/test.p5.out
Done!
```

#### Evaluating
We decided to write a script that runs evalResult.py on our predictions for all languages on a specified part, and save the output to a neat file. To evaluate F scores, just do

```
python evalAll.py p[2/3/4/5]
```

The argument after evalAll.py specifies which part to run evalResult.py for. The results would be saved in the main directory as p[2/3/4/5]\_results

## Authors
SUTD ISTD Class of 2019

[Rayson Lim 1002026](https://github.com/nosyarlin)

[Kimberlyn Loh 1002221](https://github.com/kimb3rlyn)

[Angelia Lau 1002417](https://github.com/angelialau)

