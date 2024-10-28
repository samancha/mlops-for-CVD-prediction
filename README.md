# mlops-for-CVD-prediction

Group 1 Project for ML-Ops Course at USD

#### Project requirements:

## Setup

Install all modules at the top of Group1-Final-Project-ipynb before running the notebook. It should work successfully. 

Development on the front end stream lit app, may need to install the packages in requirements.txt

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

A batch set was tested in the main notebook and you can review results. Results are stored in an S3 bucekt at : s3://{bucket}/CVDTransformBatch/

## Next steps
* integrate cloud watch to continously monitor the batch set for drift or deviation
