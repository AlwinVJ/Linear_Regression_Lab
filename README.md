# Regression Lab

**An interactive visualization tool for understanding how regression models work under the hood.**

Regression Lab is a beginner-friendly, browser-based learning tool designed to make the mathematics and mechanics of regression easier to understand through **interactive visualizations and hands-on experimentation**.

Instead of treating regression as a black box, the project lets you see how **data → predictions → errors → loss → optimization → model** are connected.

---

## What You'll Learn

Regression Lab currently focuses on **Linear Regression** and helps visualize:

* How a regression line fits data
* Model parameters (`β₀` and `β₁`)
* Predictions
* Residuals
* Mean Squared Error (MSE)
* Ordinary Least Squares (OLS)
* Gradient Descent
* Learning rate
* Parameter updates
* Model convergence

The goal is to build an accurate **mental model of what happens inside a regression algorithm**.

---

## Features

### Interactive Dataset

* Edit individual data points
* Add and remove observations
* Reset the dataset
* Experiment with different relationships between `x` and `y`

### Regression Visualization

Visualize:

* Training data
* Regression line
* Predictions
* Residuals
* Model parameters

Move the model parameters manually and immediately see how the predictions and errors change.

### Mean Squared Error

See how individual prediction errors contribute to the overall MSE.

The tool visualizes the relationship:

```text
Prediction
     ↓
Residual
     ↓
Squared Error
     ↓
Mean Squared Error
```

### Ordinary Least Squares

Explore how the best-fitting line can be calculated using the analytical OLS solution.

For simple linear regression:

```text
β₁ = Σ((xᵢ − x̄)(yᵢ − ȳ))
     ───────────────────────
       Σ(xᵢ − x̄)²

β₀ = ȳ − β₁x̄
```

The application also walks through the calculation step by step.

### Gradient Descent

Watch a regression model learn its parameters iteratively.

Each iteration performs:

```text
Predictions
     ↓
Residuals
     ↓
MSE
     ↓
Gradients
     ↓
Parameter Update
     ↓
New Model
```

You can:

* Step through one iteration at a time
* Run the learning process automatically
* Adjust the learning rate
* Change the initial parameters
* Observe convergence
* Visualize loss over iterations
* Track parameter changes

### Loss Visualization

Track how MSE changes during training and observe how gradient descent moves toward the minimum.

---

## Mathematical Foundation

The project implements the underlying calculations directly in **TypeScript**, rather than relying on machine-learning libraries.

### Linear Regression

```text
ŷ = β₀ + β₁x
```

### Residual

```text
eᵢ = yᵢ − ŷᵢ
```

### Mean Squared Error

```text
MSE = (1/n) Σeᵢ²
```

### Gradient Descent

For the intercept:

```text
∂MSE/∂β₀ = -(2/n) Σ(yᵢ − ŷᵢ)
```

For the slope:

```text
∂MSE/∂β₁ = -(2/n) Σxᵢ(yᵢ − ŷᵢ)
```

Parameter update:

```text
β = β − α × gradient
```

where `α` is the learning rate.

---

## Tech Stack

* **React**
* **TypeScript**
* **Tailwind CSS**
* Browser-side mathematical calculations
* Interactive data visualizations

No backend or external ML service is required.

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Navigate into the project:

```bash
cd <repository-name>
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL provided by Vite in your browser.

---

## Project Structure

The project separates the mathematical logic from the UI components.

```text
src/
├── algorithms/
│   ├── linearRegression.ts
│   └── gradientDescent.ts
│
├── components/
│   ├── DatasetEditor
│   ├── RegressionChart
│   ├── ModelControls
│   ├── MetricsPanel
│   ├── ResidualTable
│   └── ExplanationPanel
│
├── utils/
│   ├── calculations.ts
│   └── dataset.ts
│
└── pages/
    └── RegressionLab
```

The architecture is designed to make it easy to add additional regression algorithms in future phases.

---

## Learning Philosophy

Regression is often introduced as:

> "Fit a line to the data."

That description hides most of the interesting parts.

Regression Lab instead focuses on the chain:

```text
DATA
 ↓
MODEL
 ↓
PARAMETERS
 ↓
PREDICTIONS
 ↓
RESIDUALS
 ↓
LOSS
 ↓
GRADIENT
 ↓
PARAMETER UPDATE
 ↓
BETTER MODEL
```

The objective is to make these relationships **visible and interactive**.

---

## Roadmap

### Phase 1 — Linear Regression

* [x] Interactive dataset
* [x] Scatter plot
* [x] Regression line
* [x] Predictions
* [x] Residuals
* [x] MSE
* [x] Manual slope/intercept controls
* [x] OLS calculation
* [x] Mathematical explanations

### Phase 2 — Gradient Descent

* [ ] Gradient calculation
* [ ] Parameter updates
* [ ] Step-by-step learning
* [ ] Learning-rate controls
* [ ] Loss vs iteration
* [ ] Parameter trajectory
* [ ] Convergence visualization
* [ ] OLS vs Gradient Descent comparison

### Phase 3 — Polynomial Regression

* [ ] Polynomial feature generation
* [ ] Degree control
* [ ] Curve fitting
* [ ] Model complexity visualization
* [ ] Overfitting demonstration
* [ ] Training vs test error

### Phase 4 — Ridge Regression

* [ ] L2 regularization
* [ ] Regularization strength control
* [ ] Coefficient shrinkage visualization
* [ ] Effect of regularization on model complexity

### Phase 5 — Lasso Regression

* [ ] L1 regularization
* [ ] Coefficient shrinkage
* [ ] Feature selection visualization
* [ ] Ridge vs Lasso comparison

---

## Why This Project?

Machine-learning libraries make it extremely easy to train a regression model:

```python
model.fit(X, y)
```

But this hides what is actually happening.

This project is an attempt to make the underlying process observable and interactive so that beginners can understand **what those few lines of code are actually doing**.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

If you find a mathematical error, visualization issue, or educational improvement, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.

---

**Built for learning regression by actually seeing it happen.**
