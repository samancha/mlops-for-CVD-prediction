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

## Testing (showing metric coverage)

We can batch some data from our mock data location to create predictions that will be watched by Cloudwatch
