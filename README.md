# A numerical in a haystack: Detecting false numerical columns
#### Dani Bekman and Gil Or
This repository is submitted as the final project in the Tabular Data Science course by Dr. Amit Somech at Bar Ilan University.

We created a model to detecet whether a numerical column in a given tabular dataset is in fact categorical, ordinal or nominal, within a prediction task.

Our main code is in the main.py file, it requires a dataset path, the target column name as string and whther the prediction is a regression task (classification is default).

An example run is in the example.ipynb.

## Repository Structure:
1. The top directory - conatains the main.py file, example.iptnb, requirements.txt file and the final submission PDF reprot.
2. The dataset folder - contains the datasets we used for experiments, the code we used to fit the data to our requirements and a .md file explaining the changes we applied on the data.
3. The linear_regression_solution folder - contains the code and experiments for our linear regression baseline approach.
4. The neural_network_solution folder - contains the code and experiments for our final solution.
