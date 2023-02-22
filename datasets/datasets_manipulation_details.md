# Datasets Manipulation Details

In this file we will explain the manipulations we applied on the datasets in order to make them suitable for testing our solution. The manipulations are implemented in the edit_dataset.py code file.

The main idea was to convert a known categorical columns, nominal and ordinal, into a numeric colmuns that we could test.
The nominal categorical columns were mapped randomly to integer values, whereas the ordinal categorical columns were mapped by a pre-defined order to integer values.


### 1. Adult Dataset:
The following categorical columns were converted:
'workclass', 'marital-status', 'occupation', 'relationship', 'native-country', 'education'.

The first five were mapped randomly ('workclass', 'marital-status', 'occupation', 'relationship', 'native-country').

The 'education' column was mapped as follows: 'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6, '12th': 7,
                  				'HS-grad': 8, 'Prof-school': 9,'Assoc-acdm': 10, 'Assoc-voc': 11, 'Some-college': 12, 'Bachelors': 13
                				, 'Masters': 14, 'Doctorate': 15

In this dataset we also converted the 'income' target column to 0,1 instead of the original '<=50K', '>50K' accordingly.


### 2. Spotify Dataset:
The 'genre' column was converted randomly.


### 3. Titanic Dataset:
The 'Embarked' column was converted randomly.


### Video Games Sales Dataset:
The 'Platform' and 'Genre' columns were converted randomly.

