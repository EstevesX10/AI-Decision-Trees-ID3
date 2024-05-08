<div align="center">

# AI | Decision Trees [ID3 with Python]
</div>

<p align="center" width="100%">
    <img src="./Decision Trees - ID3/Assets/Decision_Tree.png" width="55%" height="55%" />
</p>

## Project Overview

Nowadays, **Decision Trees** represent one of the most popular **Supervised Machine Learning Algorithms**. They are commonly used in **classification problems**, but yet versatile enough to address **regression tasks** as well. The core concept behind a decision tree resides in **consecutive partitions of data** using feature-based decision-making processes that can be visually represented with a tree structure. This structure consists of nodes and leaves where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node holds a target class label.

These can vary from:

- **ID3 Algortihm**.
- **CART Algorithm**.
- **(C4.5) Algorithm**.

And even used within **ensemble models**:

- **Random Forest** (Use of Multiple Decision Trees to make a decision).
- **AdaBoost** (Use of Stumps as Weak Learners to make a decision).
- **XGBoost** (One of the best decision trees algorithms used in real world applications).

Consequently, due to their usefullness, this **Assignment #2** focuses on implementing one of these Algorithms: 

<div align="center">

> The ID3 Algorithm.
</div> 

## Project Development (Dependencies & Execution)
As a request from our professors this project was developed using a `Notebook`. Therefore if you're looking forward to test it out yourself, keep in mind to either use a **[Anaconda Distribution](https://www.anaconda.com/)** or a 3rd party software that helps you inspect and execute it.

### Dependencies

In order to install the necessary **libraries** to execute this `Project` you can either execute the following command in your `environment's terminal`:

    pip install -r requirements.txt

Or use it inside a `jupyter notebook's code cell`:

    !pip install -r requirements.txt

Another approach would be to `Create a New Anaconda Environment` with all the dependencies already installed. To do so, type:

    conda env create -f ID3.yml

### Execution
Since the Project was developed using a `Jupyter Notebook` you will need to type the following **command** in order to inspect it:

    jupyter notebook 

You can even access it via `Jupyter Lab` with:

    jupyter lab

Once the local server starts simply access it and navigate through your `Machine's Directories` until you find the folder where the **Notebook** is being held at.

<div align="right">

<sub>README.md by [Gonçalo Esteves](https://github.com/EstevesX10)</sub>
</div>