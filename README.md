
# PDF Function Approximation Models: KDE, Splines, and Neural Networks

## Project Description

This project explores multiple approaches to approximate the **probability density function (PDF)** of an unknown distribution, based only on a random sample drawn from that distribution. The methods implemented in this project include both **classical techniques** (like **Kernel Density Estimation (KDE)** and **Splines**) and **deep learning methods** (such as **Neural Networks (NN)**).

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ArmanFahradyan/density-function-approximation.git
cd density-function-approximation
pip install -r requirements.txt
```

## Input Variables

The script accepts the following input variables:

1. **`method`**: Specifies which method will be used for density function approximation. Options:
   - **KDE**: Kernel Density Estimation.
   - **Splines**: Spline-based approximation.
   - **NN**: Neural Networks.

2. **`approach`**: Specifies the specific approach within the selected method. Options:
   - For **KDE**:
     - **Kernel Name**: Choose the kernel type. Options include:
       - `"Gaussian"`, `"Epanechnikov"`, `"Cosine"`, `"Linear"`.
     - **Bandwidth Algorithm**: Select a bandwidth estimation method. Options:
       - `"Silverman"`, `"MLCV"`.
   - For **Splines**:
     - **Spline Type**: Select the type of spline approximation:
       - `"Cubic"`, `"B-Spline"`.
     - **Degree**: For B-Splines, specify the degree (integer value).
   - For **NN**:
     - **Approach**: Choose the type of neural network approach (currently, the default is `"Classic"`).

3. **`density_function`**: Defines the type of density function to approximate. Options include:
   - `"normal"`, `"uniform"`, `"gamma"`, and others (custom distributions can also be specified).

## Example Usage

Once the repository is set up, you can run the application with different configurations. Below are a few examples:

### Example 1: Using Splines for Uniform Distribution

To approximate the density function of a **uniform distribution** using **B-Splines** (degree 3), run the following command:

```bash
python main.py --method Splines --approach B-Spline --degree 3 --density_function uniform
```

### Example 2: Using KDE for Normal Distribution

To approximate the density function of a **normal distribution** using **KDE** with the **Gaussian kernel** and **Silverman bandwidth algorithm**, run:

```bash
python main.py --method KDE --approach classic --density_function normal --kernel Gaussian --bandwidth Silverman
```

### Example 3: Using Neural Networks for Gamma Distribution

To approximate the density function of a **gamma distribution** using a **classic neural network** approach:

```bash
python main.py --method NN --approach classic --density_function gamma
```

## Requirements

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```
