# mlops-for-CVD-prediction

Group 1 Project for ML-Ops Course at USD

#### Project requirements:

**Deliverable #3: Code**

1. Include a link to your team’s GitHub repository in your design document.
2. Your GitHub repository should reflect the following **eight** requirements:

* Method

  * All of your code should be stored in GitHub in a clean and professional manner. Notebooks should be stored in .ipynb format.
  * Your code should be clean, have useful comments, and only include code that builds towards the project goal.
  * Your data should be stored in S3 and documented in your GitHub repository
  * Any graphics, such as charts/graphs that help explain your data, should be included in your .ipynb files.
* ML Design

  * The codebase should be comprehensive and complete as an ML system codebase.
  * The codebase and design document should be mutually reinforcing, reflecting the parallel effort and scope of the ML system.
* Teamwork

  * All team members should contribute to the GitHub repository.
  * Commit history will be available to the instructor for review.

## Setup

Install all modules at the top of Group1-Final-Project-ipynb before running the notebook.

For the front end stream lit app, you may need to install the packages in requirements.txt

```
pip install -r requirements.txt
```

## Data

Kaggle dataset: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Features:

1. Age | Objective Feature | age | int (days)
2. Height | Objective Feature | height | int (cm) |
3. Weight | Objective Feature | weight | float (kg) |
4. Gender | Objective Feature | gender | categorical code |
5. Systolic blood pressure | Examination Feature | ap_hi | int |
6. Diastolic blood pressure | Examination Feature | ap_lo | int |
7. Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
8. Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
9. Smoking | Subjective Feature | smoke | binary |
10. Alcohol intake | Subjective Feature | alco | binary |
11. Physical activity | Subjective Feature | active | binary |
12. Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

## Testing (showing metric coverage)

We can batch some data from our mock data location to create predictions that will be watched by Cloudwatch
