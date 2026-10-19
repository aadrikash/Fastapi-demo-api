# Machine Learning Pipeline Explained: Insurance Premium Category Prediction

## 1. Project Overview

This project builds a machine learning classification model that predicts an insurance premium category based on customer information.

The model predicts one of three categories:

- `Low`
- `Medium`
- `High`

We use a **Random Forest Classifier** and a Scikit-learn `Pipeline` to combine data preprocessing and model training into one workflow.

### Technologies Used

| Library | Purpose |
|---|---|
| Pandas | Loading and manipulating tabular data |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning and preprocessing |
| OneHotEncoder | Converting categorical features into numerical features |
| ColumnTransformer | Applying different transformations to selected columns |
| Pipeline | Combining preprocessing and model training |
| RandomForestClassifier | Classification algorithm |
| train_test_split | Splitting data into training and testing sets |
| accuracy_score | Measuring prediction accuracy |
| classification_report | Generating detailed classification metrics |
| Pickle | Saving the trained model pipeline |

---

## 2. Import Required Libraries

```python
import pandas as pd

# Import the Random Forest classification algorithm.
from sklearn.ensemble import RandomForestClassifier

# Import the function used to divide data into training and testing sets.
from sklearn.model_selection import train_test_split

# Import One-Hot Encoding for categorical features.
from sklearn.preprocessing import OneHotEncoder

# Import Pipeline to combine preprocessing and model training.
from sklearn.pipeline import Pipeline

# Import ColumnTransformer to apply different operations to different columns.
from sklearn.compose import ColumnTransformer

# Import metrics for evaluating the trained model.
from sklearn.metrics import classification_report, accuracy_score

# Import NumPy for numerical operations.
import numpy as np
```

### Why do we need these libraries?

- **Pandas:** Works with tabular data using DataFrames.
- **RandomForestClassifier:** Learns patterns from the training data and predicts categories.
- **train_test_split:** Helps evaluate the model on data it did not use for training.
- **OneHotEncoder:** Converts categorical values into numerical representations.
- **ColumnTransformer:** Applies appropriate preprocessing to different feature groups.
- **Pipeline:** Keeps preprocessing and classification together.
- **accuracy_score:** Calculates the proportion of correct predictions.
- **classification_report:** Reports precision, recall, F1-score, and support.
- **NumPy:** Provides numerical operations. It is imported here, although the supplied code does not directly use it.

---

## 3. Load the Dataset

```python
# Read the CSV file into a Pandas DataFrame.
df = pd.read_csv("/content/insurance.csv")

# Display five randomly selected rows to inspect the dataset.
df.sample(5)
```

### What is a DataFrame?

A DataFrame is a two-dimensional tabular data structure containing rows and columns.

For example:

| age | weight | height | income_lpa | smoker | city | occupation | insurance_premium_category |
|---:|---:|---:|---:|---|---|---|---|
| 49 | 89.3 | 1.65 | 13.51 | False | Kota | government_job | Medium |
| 50 | 54.2 | 1.66 | 18.60 | False | Mysore | private_job | Medium |
| 62 | 99.1 | 1.50 | 1.12 | False | Mysore | retired | High |

The dataset contains customer information and the insurance premium category associated with each customer.

**Note:** `/content/insurance.csv` is a path commonly used in Google Colab. If your CSV file is stored elsewhere, replace the path with the correct location.

---

## 4. Inspect the Occupation Categories

```python
# Display all unique occupation values in the dataset.
df["occupation"].unique()
```

Example output:

```python
array([
    "retired",
    "freelancer",
    "student",
    "government_job",
    "business_owner",
    "unemployed",
    "private_job"https://github.com/campusx-official/fastapi-demo-api/tree/main
], dtype=object)
```

### What does `unique()` do?

It returns the distinct values found in a column.

This helps identify categorical features that need to be converted into numerical representations before being used by the model.

For example, `occupation` contains text rather than numerical values.

A machine learning algorithm generally needs a numerical representation of these categories.

We will use **One-Hot Encoding** to accomplish this.

---

# Part 1: Feature Engineering

## 5. Create a Copy of the Dataset

```python
# Create a separate copy so that feature engineering does not modify the original DataFrame.
df_feat = df.copy()
```

### What is feature engineering?
https://github.com/campusx-official/fastapi-demo-api/tree/main
Feature engineering means creating or transforming input features to help a machine learning model learn useful patterns.

Instead of using only the original columns, we derive additional features from existing data.

In this project, we create:

1. BMI
2. Age group
3. Lifestyle risk
4. City tier

These features are then used to train the model.

---

## 6. Feature 1: Calculate BMI

```python
# Calculate Body Mass Index using weight in kilograms and height in metres.
df_feat["bmi"] = df_feat["weight"] / (df_feat["height"] ** 2)
```

### Formula

BMI is calculated as:

\[
BMI = \frac{Weight\ (kg)}{Height\ (m)^2}
\]

For example, if a customer weighs 70 kg and has a height of 1.75 metres:

\[
BMI = \frac{70}{1.75^2}
\]

\[
BMI \approx 22.86
\]

### Understanding the code

- `df_feat["weight"]` retrieves the weight column.
- `df_feat["height"]` retrieves the height column.
- `** 2` squares the height.
- `/` divides weight by squared height.
- `df_feat["bmi"] =` stores the calculated values in a new column.

Pandas performs this calculation for every row.

**Why create BMI?**

BMI combines weight and height into a single feature that can represent a person's body size more meaningfully than either measurement alone.

BMI is a simplified measure and does not independently determine a person's health or insurance risk.

---

## 7. Feature 2: Create Age Groups

```python
# Define a function that converts a numerical age into an age-group category.
def age_group(age):

    # Customers younger than 25 belong to the young group.
    if age < 25:
        return "young"

    # Customers aged 25 to less than 45 belong to the adult group.
    elif age < 45:
        return "adult"

    # Customers aged 45 to less than 60 belong to the middle-aged group.
    elif age < 60:
        return "middle_aged"

    # Customers aged 60 or above belong to the senior group.
    return "senior"


# Apply the function to every value in the age column.
df_feat["age_group"] = df_feat["age"].apply(age_group)
```

### Age-group rules

| Age | Result |
|---|---|
| Less than 25 | `young` |
| 25 to 44 | `adult` |
| 45 to 59 | `middle_aged` |
| 60 or above | `senior` |

### What does `.apply()` do?

The following expression:

```python
df_feat["age"].apply(age_group)
```

passes each age value to the `age_group()` function.

The function returns a category for each customer, and the results are stored in the new `age_group` column.

### Why create age groups?

Age groups can make broad age-related patterns easier for a model to learn.

However, grouping ages also removes some information because two customers with different ages may belong to the same group.

---

## 8. Feature 3: Calculate Lifestyle Risk

```python
# Define a function that uses smoking status and BMI to assign a lifestyle-risk category.
def lifestyle_risk(row):

    # Assign high risk when the customer smokes and has a BMI above 30.
    if row["smoker"] and row["bmi"] > 30:
        return "high"

    # Assign medium risk when the customer smokes or has a BMI above 27.
    elif row["smoker"] or row["bmi"] > 27:
        return "medium"

    # Assign low risk when neither condition is satisfied.
    else:
        return "low"


# Apply the function to each row of the DataFrame.
df_feat["lifestyle_risk"] = df_feat.apply(lifestyle_risk, axis=1)
```

### Understanding the conditions

| Smoker | BMI | Assigned category |
|---|---:|---|
| True | Above 30 | `high` |
| True | 30 or below | `medium` |
| False | Above 30 | `medium` |
| False | 27 or below | `low` |
| False | Above 27 and up to 30 | `medium` |

### What does `axis=1` mean here?

```python
df_feat.apply(lifestyle_risk, axis=1)
```

- `axis=1` applies the function to each **row**.
- `axis=0` would apply a function to each column.

Our function needs both `smoker` and `bmi` from the same customer. Therefore, it processes one row at a time.

### Important distinction

`lifestyle_risk` is a feature created using rules that we defined ourselves.

The categories `high`, `medium`, and `low` are not medical diagnoses or independently verified insurance-risk assessments.

---

## 9. Feature 4: Categorise Cities into Tiers

First, define the city lists.

```python
# List cities assigned to Tier 1.
tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune"
]


# List cities assigned to Tier 2.
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna",
    "Ranchi", "Visakhapatnam", "Coimbatore", "Bhopal",
    "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur",
    "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun",
    "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram",
    "Ludhiana", "Nashik", "Allahabad", "Udaipur",
    "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada",
    "Tiruchirappalli", "Bhavnagar", "Gwalior", "Dhanbad",
    "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur",
    "Asansol", "Siliguri"
]
```

Now create the function.

```python
# Define a function that assigns a numerical tier to each city.
def city_tier(city):

    # Return tier 1 if the city appears in the Tier 1 list.
    if city in tier_1_cities:
        return 1

    # Return tier 2 if the city appears in the Tier 2 list.
    elif city in tier_2_cities:
        return 2

    # Assign tier 3 to cities that do not appear in either list.
    else:
        return 3


# Apply the function to create the city_tier column.
df_feat["city_tier"] = df_feat["city"].apply(city_tier)
```

### How does it work?

The function checks each city against the two lists.

- A city in `tier_1_cities` receives `1`.
- A city in `tier_2_cities` receives `2`.
- Any other city receives `3`.

### Why transform city names into tiers?

The transformation replaces individual city names with a smaller set of numerical categories.

For example:

| City | City tier |
|---|---:|
| Bangalore | 1 |
| Mysore | 2 |
| A city not included in either list | 3 |

The tier assignments follow the lists defined in this project. They should not be interpreted as an authoritative or universally accepted classification of Indian cities.

---

## 10. Inspect the Engineered Features

```python
# Remove the original columns that will not be used directly by this model.
# Display the selected engineered features and the target column.
df_feat.drop(
    columns=["age", "weight", "height", "smoker", "city"]
)[
    [
        "income_lpa",
        "occupation",
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "insurance_premium_category"
    ]
].sample(5)
```

### Why remove the original columns?

We have derived new features from several original columns:

| Original columns | Derived feature |
|---|---|
| `weight`, `height` | `bmi` |
| `age` | `age_group` |
| `smoker`, `bmi` | `lifestyle_risk` |
| `city` | `city_tier` |

We then select the engineered features for the model.

**Important:** The `drop()` call above does not permanently modify `df_feat` because its result is not assigned back to the DataFrame.

---

# Part 2: Select Features and Target

## 11. Define X and y

```python
# Select the input features that the model will use to make predictions.
X = df_feat[
    [
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "income_lpa",
        "occupation"
    ]
]


# Select the target column that the model must learn to predict.
y = df_feat["insurance_premium_category"]
```

### What is X?

`X` contains the input features used by the model.

In this project:

- `bmi`
- `age_group`
- `lifestyle_risk`
- `city_tier`
- `income_lpa`
- `occupation`

### What is y?

`y` contains the correct output category for each customer.

The target is:

```python
insurance_premium_category
```

Its values include `Low`, `Medium`, and `High`.

### Why separate X and y?

A supervised machine learning model learns a relationship between input features (`X`) and known target values (`y`).

During prediction, the model receives input features and estimates the target category.

### Inspect the inputs and target

```python
# Display the input features.
X

# Display the target values.
y
```

For the supplied dataset, there are 100 rows and six input features.

---

# Part 3: Preprocessing with One-Hot Encoding

## 12. Identify Categorical and Numerical Features

```python
# List the categorical features that contain categories or labels.
categorical_features = [
    "age_group",
    "lifestyle_risk",
    "occupation",
    "city_tier"
]


# List the numerical features that contain numerical measurements.
numeric_features = [
    "bmi",
    "income_lpa"
]
```

### Why separate these features?

Different kinds of features may need different preprocessing.

For example:

- `occupation` contains text labels.
- `age_group` contains text labels.
- `lifestyle_risk` contains text labels.
- `city_tier` contains numerical category codes.
- `bmi` contains numerical measurements.
- `income_lpa` contains numerical measurements.

We will apply One-Hot Encoding to the categorical features and pass the numerical features through unchanged.

Although `city_tier` is stored as a number, it represents categories rather than a continuous measurement. Treating it as categorical avoids implying that the numerical distance between tiers is a meaningful quantity for the model.

---

## 13. What Is One-Hot Encoding?

Machine learning estimators generally require numerical input.

However, a categorical feature such as `occupation` contains text:

```text
student
retired
private_job
government_job
```

One-Hot Encoding creates a separate binary column for each category learned during fitting.

For example, a simplified encoding might look like this:

| occupation | occupation_student | occupation_retired | occupation_private_job |
|---|---:|---:|---:|
| student | 1 | 0 | 0 |
| retired | 0 | 1 | 0 |
| private_job | 0 | 0 | 1 |

A value of `1` indicates the category represented by that column.

A value of `0` indicates that the category is not present.

The actual encoded columns depend on the categories found in the training data.

---

## 14. Create the ColumnTransformer

```python
# Import the tools required for preprocessing.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Create a transformer that applies the appropriate operation to each feature group.
preprocessor = ColumnTransformer(
    transformers=[

        # Apply One-Hot Encoding to the categorical features.
        ("cat", OneHotEncoder(), categorical_features),

        # Keep numerical features unchanged.
        ("num", "passthrough", numeric_features)

    ]
)
```

### What is ColumnTransformer?

`ColumnTransformer` allows us to apply different transformations to different columns.

In this project:

```python
("cat", OneHotEncoder(), categorical_features)
```

means:

1. `"cat"` is the name assigned to this transformation.
2. `OneHotEncoder()` is the transformation to apply.
3. `categorical_features` identifies the columns to transform.

The numerical transformation is:

```python
("num", "passthrough", numeric_features)
```

Here:

- `"num"` is the transformation name.
- `"passthrough"` means keep the selected columns unchanged.
- `numeric_features` identifies the columns that should pass through.

### Why not encode every column?

Numerical measurements such as BMI and income can remain numerical.

Categorical labels such as occupation need to be converted into a numerical representation.

Separating the operations keeps the preprocessing logic organised and consistent.

---

# Part 4: Build the Machine Learning Pipeline

## 15. What Is a Pipeline?

A Scikit-learn `Pipeline` combines multiple machine learning steps into a single object.

Our pipeline has two steps:

1. `preprocessor`: transforms the input features.
2. `classifier`: learns patterns and predicts the target category.

The data flows through these steps in order.

```text
Raw input features
        |
        v
ColumnTransformer
        |
        +--> OneHotEncoder for categorical features
        |
        +--> Pass numerical features through unchanged
        |
        v
Processed numerical features
        |
        v
RandomForestClassifier
        |
        v
Predicted insurance premium category
```

## 16. Create the Pipeline

```python
# Import the Pipeline class.
from sklearn.pipeline import Pipeline

# Import the Random Forest classification algorithm.
from sklearn.ensemble import RandomForestClassifier


# Combine preprocessing and classification into one pipeline.
pipeline = Pipeline(
    steps=[

        # First, preprocess the input features.
        ("preprocessor", preprocessor),

        # Next, train a Random Forest classifier on the processed features.
        ("classifier", RandomForestClassifier(random_state=42))

    ]
)
```

### Understanding the code

#### `Pipeline`

```python
Pipeline(steps=[...])
```

Creates a sequence of operations that Scikit-learn executes in order.

#### `"preprocessor"`

```python
("preprocessor", preprocessor)
```

Uses the `ColumnTransformer` defined earlier.

It converts categorical inputs into numerical features and passes through the selected numerical inputs.

#### `"classifier"`

```python
("classifier", RandomForestClassifier(random_state=42))
```

Creates the classification model.

The name `"classifier"` identifies the pipeline step. The `RandomForestClassifier` is the actual machine learning algorithm.

#### What is Random Forest?

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

For classification, the trees collectively determine the predicted class.

It can model nonlinear relationships and interactions between features.

#### What does `random_state=42` do?

It fixes the random seed for the classifier so its randomised operations can be reproduced under the same conditions.

The value `42` is arbitrary; other fixed integer values can also be used.

**Important:** Creating the pipeline does not train the model. Training occurs when we call `pipeline.fit()`.

---

# Part 5: Split the Data

## 17. Create Training and Testing Sets

```python
# Import the train-test splitting function.
from sklearn.model_selection import train_test_split


# Divide the input features and target into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Input features.
    y,                  # Target labels.
    test_size=0.2,      # Reserve 20% of the data for testing.
    random_state=1      # Make the split reproducible.
)
```

### Why split the data?

We want to evaluate how the model performs on observations it did not use for training.

With 100 rows and `test_size=0.2`, the split produces:

| Dataset | Number of rows |
|---|---:|
| Training set | 80 |
| Testing set | 20 |
| Total | 100 |

The training set is used to fit the pipeline.

The testing set is used to evaluate predictions.

### What does `random_state=1` do here?

It makes the random train-test split reproducible.

Notice that this split uses `random_state=1`, while the Random Forest uses `random_state=42`. They control randomness in two different operations.

---

# Part 6: Train the Model

## 18. Fit the Pipeline

```python
# Fit the complete pipeline using the training features and target labels.
pipeline.fit(X_train, y_train)
```

### What happens inside `fit()`?

The pipeline performs the following operations in order:

**Step 1: Fit the preprocessor**

The `ColumnTransformer` learns the categorical values present in the training data.

The `OneHotEncoder` learns the categories it will encode.

The numerical columns are passed through unchanged.

**Step 2: Transform the training features**

The categorical columns are converted into numerical columns.

The numerical features remain unchanged.

**Step 3: Train the Random Forest**

The transformed training features and `y_train` are passed to the classifier.

The Random Forest learns patterns associated with the target categories.

### Why use a pipeline for training?

It keeps preprocessing and model training together.

It also helps prevent data leakage because the preprocessing steps are fitted on the training data rather than the test data.

---

# Part 7: Make Predictions

## 19. Predict the Test Set

```python
# Use the fitted pipeline to predict the target category for each test observation.
y_pred = pipeline.predict(X_test)
```

### What happens during prediction?

The pipeline automatically:

1. Receives the test features.
2. Applies the already-fitted preprocessing steps.
3. Converts categorical features using the categories learned during training.
4. Passes the transformed features to the trained Random Forest.
5. Returns the predicted insurance premium categories.

We do not need to call `OneHotEncoder` separately.

This is one of the main advantages of using a pipeline.

---

# Part 8: Evaluate the Model

## 20. Calculate Accuracy

```python
# Import the accuracy metric.
from sklearn.metrics import accuracy_score


# Compare the actual test labels with the predicted labels.
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy.
print(accuracy)
```

The supplied code produced:

```text
0.9
```

### What does 0.9 mean?

Accuracy is the proportion of correct predictions.

\[
Accuracy = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
\]

For 20 test observations, an accuracy of `0.9` corresponds to:

\[
Accuracy = \frac{18}{20} = 0.90
\]

That means 18 of the 20 predictions were correct for this particular test split.

### Is 90% accuracy enough?

Not by itself.

The dataset contains only 100 rows, so the test set contains only 20 observations.

A small test set can produce an unstable estimate of real-world performance. More evaluation is needed before concluding that the model will perform well on new customers.

---

## 21. Generate a Classification Report

Although `classification_report` is imported in the original code, it is not called there.

To use it, add:

```python
# Display detailed classification metrics for the test predictions.
print(classification_report(y_test, y_pred))
```

The report includes:

| Metric | Meaning |
|---|---|
| Precision | Of the observations predicted as a class, how many actually belong to that class? |
| Recall | Of the observations that actually belong to a class, how many did the model identify? |
| F1-score | The harmonic mean of precision and recall |
| Support | Number of actual test observations belonging to each class |

For a multiclass problem, the report provides metrics for each class, along with summary averages.

Because the test set is small, some classes may have very few test observations. Interpret the results accordingly.

---

## 22. Inspect Test Features

```python
# Display five randomly selected rows from the test feature set.
X_test.sample(5)
```

This helps us inspect the inputs that were held out from training.

To compare actual and predicted categories, you can create a table:

```python
# Create a DataFrame that compares actual and predicted target categories.
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

# Display five randomly selected comparisons.
comparison.sample(5)
```

The index helps match each prediction to its original observation.

---

# Part 9: Save the Trained Pipeline

## 23. Save the Model Using Pickle

```python
# Import Python's built-in serialization library.
import pickle


# Define the filename used to store the trained pipeline.
pickle_model_path = "model.pkl"


# Open the file in binary-write mode.
with open(pickle_model_path, "wb") as f:

    # Save the fitted pipeline, including preprocessing and the classifier.
    pickle.dump(pipeline, f)
```

### What is Pickle?

Pickle is a Python library for serializing Python objects.

Serialization converts an object into a representation that can be saved to a file and later loaded back into Python.

### Why save the entire pipeline?

We save:

- The fitted `ColumnTransformer`.
- The fitted `OneHotEncoder`.
- The trained `RandomForestClassifier`.

This means we do not need to manually recreate and fit the preprocessing steps when using the saved model.

### What does `"wb"` mean?

- `w` means write.
- `b` means binary mode.

Pickle files are written in binary mode.

### What does `pickle.dump()` do?

```python
pickle.dump(pipeline, f)
```

Writes the fitted pipeline object into the opened file.

The resulting file is named:

```text
model.pkl
```

The file is saved in the current working directory unless you specify another path.

**Security note:** Only load pickle files from trusted sources. Loading a malicious pickle can execute arbitrary code.

---

## 24. Load the Saved Pipeline

You can load the saved pipeline later:

```python
# Import the pickle library.
import pickle


# Open the saved model file in binary-read mode.
with open("model.pkl", "rb") as f:

    # Deserialize the saved pipeline.
    loaded_pipeline = pickle.load(f)
```

You can then use it to make predictions:

```python
# Predict categories using the loaded pipeline.
loaded_predictions = loaded_pipeline.predict(X_test)

# Display the predictions.
print(loaded_predictions)
```

The saved pipeline applies the fitted preprocessing steps before making predictions.

The input data must contain the same feature columns expected by the pipeline.

For reliable deployment, keep the model's feature names, feature types, and preprocessing logic consistent between training and prediction.

---

# Part 10: How the Complete Workflow Fits Together

```text
                 insurance.csv
                      |
                      v
                 Load with Pandas
                      |
                      v
             Feature Engineering
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
         BMI       Age Group    Lifestyle Risk
                      |
                      v
                  City Tier
                      |
                      v
             Select X and target y
                      |
                      v
              Train-Test Split
                      |
             +--------+--------+
             |                 |
             v                 v
        Training Data       Test Data
             |
             v
        Pipeline.fit()
             |
             v
        ColumnTransformer
             |
             +-------------------------+
             |                         |
             v                         v
       One-Hot Encoding          Numerical Passthrough
             |                         |
             +------------+------------+
                          |
                          v
                  Random Forest
                          |
                          v
                   Trained Pipeline
                          |
              +-----------+-----------+
              |                       |
              v                       v
         Test Predictions          model.pkl
              |
              v
          Evaluation
              |
              v
       Accuracy and Metrics
```

---

# Part 11: Important Concepts to Remember

| Concept | Explanation |
|---|---|
| Feature engineering | Creating useful input features from existing data |
| Feature (`X`) | Input information used to make predictions |
| Target (`y`) | Output the model learns to predict |
| Categorical feature | A feature representing categories or labels |
| Numerical feature | A feature representing a numerical value |
| One-Hot Encoding | Converts categorical values into binary columns |
| ColumnTransformer | Applies different transformations to selected columns |
| Pipeline | Combines preprocessing and model training into one object |
| Random Forest | An ensemble of decision trees used for classification |
| Training set | Data used to fit the model |
| Testing set | Data held out for evaluation |
| `fit()` | Learns preprocessing parameters and model patterns from training data |
| `predict()` | Produces predictions for supplied inputs |
| Accuracy | Fraction of predictions that are correct |
| Pickle | Saves and reloads Python objects |

---

# Part 12: Complete Code

The following combines the essential steps from the supplied implementation into one runnable script.

```python
# Import libraries for data handling and model building.
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

# Read the dataset.
df = pd.read_csv("/content/insurance.csv")


# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

# Copy the original DataFrame.
df_feat = df.copy()


# Feature 1: Calculate BMI.
df_feat["bmi"] = df_feat["weight"] / (df_feat["height"] ** 2)


# Feature 2: Create age groups.
def age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


df_feat["age_group"] = df_feat["age"].apply(age_group)


# Feature 3: Assign lifestyle-risk categories.
def lifestyle_risk(row):
    if row["smoker"] and row["bmi"] > 30:
        return "high"
    elif row["smoker"] or row["bmi"] > 27:
        return "medium"
    else:
        return "low"


df_feat["lifestyle_risk"] = df_feat.apply(lifestyle_risk, axis=1)


# Feature 4: Define city tiers.
tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune"
]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna",
    "Ranchi", "Visakhapatnam", "Coimbatore", "Bhopal",
    "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur",
    "Raipur", "Amritsar", "Varanasi", "Agra", "Dehradun",
    "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram",
    "Ludhiana", "Nashik", "Allahabad", "Udaipur",
    "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada",
    "Tiruchirappalli", "Bhavnagar", "Gwalior", "Dhanbad",
    "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur",
    "Asansol", "Siliguri"
]


def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3


df_feat["city_tier"] = df_feat["city"].apply(city_tier)


# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

# Select the features used by the model.
X = df_feat[
    [
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "income_lpa",
        "occupation"
    ]
]


# Select the target to predict.
y = df_feat["insurance_premium_category"]


# --------------------------------------------------
# 4. DEFINE PREPROCESSING
# --------------------------------------------------

# Identify categorical features.
categorical_features = [
    "age_group",
    "lifestyle_risk",
    "occupation",
    "city_tier"
]


# Identify numerical features.
numeric_features = [
    "bmi",
    "income_lpa"
]


# Create the preprocessing transformer.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)


# --------------------------------------------------
# 5. CREATE THE PIPELINE
# --------------------------------------------------

# Combine preprocessing and the classifier.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)


# --------------------------------------------------
# 6. SPLIT THE DATA
# --------------------------------------------------

# Reserve 20% of the observations for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)


# --------------------------------------------------
# 7. TRAIN THE MODEL
# --------------------------------------------------

# Fit the complete pipeline on the training data.
pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 8. PREDICT AND EVALUATE
# --------------------------------------------------

# Generate predictions for the test set.
y_pred = pipeline.predict(X_test)


# Calculate and display accuracy.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Display precision, recall, F1-score, and support.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. SAVE THE TRAINED PIPELINE
# --------------------------------------------------

# Save preprocessing and the trained classifier together.
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)


print("\nTrained pipeline saved as model.pkl")
```

**Note:** In this combined version, `OneHotEncoder(handle_unknown="ignore")` is a small robustness improvement over your original `OneHotEncoder()`. It allows the encoder to handle previously unseen categories during prediction without raising an unknown-category error.

Also, the complete script assumes the input dataset has the columns shown in your example and that the CSV path is correct.

---

## Final Takeaway

Your project follows a standard supervised machine learning workflow:

1. Load the dataset.
2. Engineer useful features.
3. Separate input features from the target.
4. Identify categorical and numerical columns.
5. Preprocess the categorical columns with One-Hot Encoding.
6. Combine preprocessing and classification in a pipeline.
7. Split the data into training and testing sets.
8. Train the model.
9. Evaluate its predictions.
10. Save the trained pipeline for reuse.

**The most important concept is that the saved pipeline contains both the fitted preprocessing steps and the trained classifier.** This makes it easier to use the same transformations consistently when you later serve predictions through FastAPI.

One final practical note: the example uses only 100 rows, so treat the reported 90% accuracy as an initial learning result, not proof of reliable real-world insurance prediction.



## My Doubt why are we writing same logic cod in app.py and fastapi.ipynb for city_tier and lifestyle_risk ?

The reason you see the same logic for city_tier and lifestyle_risk in both Fastapi_ml_model.ipynb and app.py is that the notebook prepares the data for training, whereas the FastAPI application prepares new data for prediction.

options we can do:
although we can write the logic function in seperate file and call them in app.py and juypter and one more way we can also add it in the pipeline as well.