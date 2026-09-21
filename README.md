Regression Lab

An interactive, browser-based learning environment for understanding regression algorithms from the mathematics to the underlying optimization process.

Regression Lab is designed for beginners who want to understand how regression models actually work, rather than treating machine-learning libraries as black boxes.

Instead of simply calling LinearRegression() or Ridge(), the application lets you interact with the data, equations, parameters, loss functions, optimization steps, regularization, and model behavior through visualizations.

What You'll Learn

Regression Lab currently covers:

Linear Regression

Gradient Descent

Polynomial Regression

Ridge Regression

Lasso Regression

Core regression concepts and mathematical foundations

The learning flow is designed to move from simple ideas to increasingly sophisticated models:

Data
  ↓
Linear Regression
  ↓
Predictions & Residuals
  ↓
Mean Squared Error
  ↓
Ordinary Least Squares
  ↓
Gradient Descent
  ↓
Polynomial Feature Expansion
  ↓
Overfitting
  ↓
Ridge Regression
  ↓
L2 Regularization
  ↓
Lasso Regression
  ↓
L1 Regularization & Feature Selection

Key Features

Linear Regression

Explore the fundamentals of fitting a straight line to data.

Editable datasets

Scatter plots

Manual slope and intercept controls

Prediction visualization

Residuals

Mean Squared Error (MSE)

Ordinary Least Squares (OLS)

Step-by-step parameter calculations

Mathematical equation → Python code explanations

The basic model is:

ŷ = β₀ + β₁x

Gradient Descent

Understand how a model can iteratively find better parameters.

The lab visualizes:

Initial parameter values

Predictions

Errors

MSE

Gradients

Parameter updates

Learning rate

Iterations

Loss progression

Parameter trajectories

Loss landscapes

Convergence

The optimization process is demonstrated using the underlying mathematics rather than a machine-learning library.

Polynomial Regression

Understand how linear regression can model nonlinear relationships through feature expansion.

For example:

ŷ = β₀ + β₁x + β₂x² + β₃x³ + ...

Explore:

Polynomial degree

Feature expansion

Curved regression models

Underfitting

Good fit

Overfitting

Training vs. test error

Model complexity

Coefficients

Polynomial feature tables

A key concept demonstrated by the application is:

Polynomial regression is still linear in its coefficients; the input features are transformed into polynomial features.

Ridge Regression

Learn how L2 regularization controls model complexity.

Ridge minimizes an objective of the form:

MSE + λ Σ βⱼ²

The application demonstrates:

Regularization strength λ

OLS vs. Ridge

Coefficient shrinkage

L2 penalty

Ridge objective

Coefficient paths

Training/test error

Polynomial models with regularization

Feature scaling

Underfitting caused by excessive regularization

The intercept is treated separately from the regularized coefficients.

Lasso Regression

Learn how L1 regularization can produce sparse models.

Lasso minimizes:

MSE + λ Σ |βⱼ|

The lab demonstrates:

L1 penalty

Coefficient shrinkage

Exact zero coefficients

Feature selection

Sparsity

Coordinate descent

Soft-thresholding

Objective function progression

Active feature counts

Coefficient paths

Correlated-feature behavior

Ridge vs. Lasso comparison

The soft-thresholding operation is visualized to show why Lasso can drive coefficients exactly to zero.

Mathematics Behind the Models

Regression Lab intentionally exposes the mathematics behind each algorithm.

Linear Regression

ŷᵢ = β₀ + β₁xᵢ

Residual

eᵢ = yᵢ - ŷᵢ

Mean Squared Error

MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Ordinary Least Squares

For simple linear regression:

β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²

β₀ = ȳ - β₁x̄

Gradient Descent

β = β - α∇L(β)

where:

β = model parameters

α = learning rate

∇L(β) = gradient of the loss function

Ridge

Loss = MSE + λ Σβⱼ²

Lasso

Loss = MSE + λ Σ|βⱼ|

These equations are connected directly to the visual behavior of the models.

Optimization Techniques

The application does not treat all models as if they were optimized in the same way.

Gradient Descent

Used to demonstrate iterative optimization:

1. Make predictions
2. Calculate errors
3. Calculate loss
4. Calculate gradients
5. Update parameters
6. Repeat

Coordinate Descent

Used to explain Lasso optimization:

1. Select a coefficient
2. Calculate its partial residual
3. Compute the coordinate update
4. Apply soft-thresholding
5. Move to the next coefficient
6. Repeat until convergence

This distinction is important because the L1 penalty introduces a non-differentiable point at zero.

Learning Philosophy

Regression Lab is built around a visual, experiment-driven learning approach.

Instead of:

model.fit(X, y)

and stopping there, the application encourages you to ask:

What is the model predicting?

Where do the errors come from?

How is MSE calculated?

Why is one line better than another?

How does OLS find the best parameters?

What does a gradient actually represent?

What happens when the learning rate changes?

Why does increasing polynomial degree create overfitting?

What does regularization actually penalize?

Why does Ridge shrink coefficients?

Why can Lasso make coefficients exactly zero?

What does λ really control?

The goal is to build mechanical understanding, not just API familiarity.

Mathematics → Python

Each learning module contains "From mathematics to code" sections connecting the equations to beginner-friendly Python representations.

For example:

Mathematics

ŷ = β₀ + β₁x

Python

prediction = intercept + slope * x

This helps connect:

Mathematical notation
        ↓
Algorithmic idea
        ↓
Python implementation
        ↓
Visual behavior

The application's actual frontend implementation remains TypeScript/React; the Python snippets are educational examples for learners.

Technology Stack

Frontend

React

TypeScript

TanStack Start / TanStack Router

Vite

Tailwind CSS

Visualization

Recharts

Custom React visualization components

Interactive controls and charts

UI

Radix UI

Lucide React

Tailwind-based styling

Architecture

The mathematical and optimization logic is implemented directly in the browser using TypeScript.

No external machine-learning library is required for the core regression demonstrations.

Architecture

The project separates mathematical algorithms, optimization methods, reusable visualizations, pages, and utility functions.

## Project Structure

```text
src/
├── algorithms/
│   ├── linearRegression.ts
│   ├── gradientDescent.ts
│   ├── polynomialRegression.ts
│   ├── ridgeRegression.ts
│   └── lassoRegression.ts
│
├── optimization/
│   ├── coordinateDescent.ts
│   └── softThresholding.ts
│
├── components/
│   ├── RegressionChart.tsx
│   ├── DatasetEditor.tsx
│   ├── CoefficientChart.tsx
│   ├── CoefficientPathChart.tsx
│   ├── LossLandscape.tsx
│   ├── PolynomialChart.tsx
│   ├── RegularizationControl.tsx
│   ├── SoftThresholdVisualization.tsx
│   └── ...
│
├── pages/
│   ├── HomePage.tsx
│   ├── RegressionLab.tsx
│   ├── LinearRegressionPage.tsx
│   ├── GradientDescentLab.tsx
│   ├── PolynomialRegression.tsx
│   ├── RidgeRegression.tsx
│   ├── LassoRegression.tsx
│   └── ConceptsPage.tsx
│
├── utils/
│   ├── calculations.ts
│   ├── dataset.ts
│   ├── metrics.ts
│   ├── polynomialFeatures.ts
│   ├── regularization.ts
│   └── scaling.ts
│
└── routes/
    ├── index.tsx
    ├── playground.tsx
    ├── linear-regression.tsx
    ├── gradient-descent.tsx
    ├── polynomial-regression.tsx
    ├── ridge-regression.tsx
    ├── lasso-regression.tsx
    └── concepts.tsx
```

The application is organized as a learning journey:

Module

Main Concept

Playground

Interactive regression experimentation

Linear Regression

Fitting a straight line

Gradient Descent

Iterative optimization

Polynomial Regression

Feature expansion and nonlinear curves

Ridge Regression

L2 regularization

Lasso Regression

L1 regularization and sparsity

Concepts

Mathematical foundations and terminology

Running Locally

Prerequisites

Install:

Node.js

npm

Check your versions:

node --version
npm --version

Clone the repository

git clone <your-repository-url>
cd <repository-name>

Install dependencies

npm install

Start the development server

npm run dev

Open the local URL shown by Vite in your browser.

Available Scripts

Development

npm run dev

Starts the Vite development server.

Production Build

npm run build

Creates a production build.

Development Build

npm run build:dev

Creates a development-mode production build.

Preview

npm run preview

Serves the production build locally.

Lint

npm run lint

Runs ESLint across the project.

Format

npm run format

Formats the project using Prettier.

Design Principles

Regression Lab follows several principles:

1. No Black Boxes

The application should expose the mechanics behind the algorithm whenever possible.

2. Mathematics First

Every major algorithm is connected to its mathematical formulation.

3. Visualization Over Abstraction

Parameters, errors, coefficients, losses, and optimization behavior are visualized rather than hidden.

4. Experimentation

Learners can change parameters and immediately observe how the model behaves.

5. Progressive Complexity

The application moves from:

Simple Linear Regression
        ↓
Optimization
        ↓
Polynomial Features
        ↓
Regularization
        ↓
Sparse Models

6. Beginner-Friendly Explanations

Technical concepts are introduced incrementally while preserving mathematical accuracy.

What This Project Is Not

Regression Lab is an educational visualization environment.

It is not intended to replace production machine-learning frameworks such as:

scikit-learn

XGBoost

LightGBM

PyTorch

TensorFlow

The purpose is to understand what happens underneath high-level machine-learning APIs.

Learning Outcomes

After working through the application, a learner should be able to explain:

What supervised regression means

What features and targets are

How a linear model makes predictions

What slope and intercept represent

What residuals are

Why MSE is used

How OLS finds parameters

What a gradient represents

How gradient descent updates parameters

How learning rate affects optimization

Why polynomial regression can model curves

What feature expansion means

What underfitting and overfitting are

Why Ridge uses an L2 penalty

Why Lasso uses an L1 penalty

How regularization affects coefficients

Why Lasso can perform feature selection

What soft-thresholding does

Why feature scaling matters

How Ridge and Lasso differ

Roadmap

Future improvements may include:

Multiple Linear Regression

Interactive feature engineering

More optimization visualizations

Elastic Net Regression

Cross-validation

Bias-variance demonstrations

Additional datasets

Interactive quizzes and learning checkpoints

More model-comparison experiments

Expanded mathematical derivations

Additional Python implementation examples

Project Status

Current status: Core regression learning modules completed.

Implemented learning modules:

Linear Regression

Gradient Descent

Polynomial Regression

Ridge Regression

Lasso Regression

Concepts / mathematical explanations

Interactive visualizations

Mathematics → Python learning sections

Contributing

Contributions are welcome, particularly improvements that make the mathematical concepts more accurate, intuitive, and interactive.

A useful contribution should ideally:

Preserve mathematical correctness.

Improve the learner's understanding.

Avoid unnecessary abstraction.

Include clear explanations for new concepts.

Keep the interactive visualizations responsive and understandable.

License

Add the license appropriate for your repository.

If this project is intended to be open source, consider using the MIT License.

Acknowledgements

Built as a hands-on learning project to make regression algorithms easier to understand through:

Mathematics

Interactive visualization

Experimentation

Algorithmic implementations

Beginner-friendly explanations

Author

Alwin V J

Built as part of a practical journey into Machine Learning, Data Science, and AI.

⭐ If You Find This Useful

If Regression Lab helps you understand regression concepts more clearly, consider starring the repository and sharing it with other learners exploring Machine Learning.