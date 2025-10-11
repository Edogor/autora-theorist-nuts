# Nuts Theorist

The Nuts Theorist is an automated equation discovery tool for the AutoRA framework. It uses a machine learning technique called **Genetic Programming** to find a mathematical formula that best explains a given set of experimental data.

Unlike "black-box" models, the goal of the Nuts Theorist is **Symbolic Regression**—its output is not just a set of predictions, but a simple, human-readable equation.

---
## The Core Idea: Evolving Equations

Imagine you want to breed the fastest racehorse. You would start with a diverse group of horses, race them, and identify the top performers. Then, you'd breed these top performers, combining their fast traits to produce a new generation of offspring. By repeating this process of selection and reproduction over many generations, you would gradually evolve a champion racehorse.

The Nuts Theorist does the exact same thing, but with mathematical equations instead of horses. It starts with a random pool of formulas, "races" them to see how well they fit the data, and then "breeds" the best ones to create a new generation of improved formulas. This evolutionary process continues until it finds a champion equation that is both simple and accurate.





---
## How It Works: The Evolutionary Cycle

The theorist's discovery process is a cycle that mimics natural selection.

### 1. Initialization
The process begins by creating a large, random population of candidate equations. Internally, these are represented as tree structures that are easy for the algorithm to modify.

### 2. Evaluation (Fitness)
Each equation in the population is scored to determine its "fitness." This score is a balance of two key factors:
- **Accuracy:** How well the equation's predictions match the actual experimental data, measured by Mean Squared Error (MSE).
- **Simplicity:** A **complexity penalty** is added to the score. This penalizes longer, more complicated equations, pushing the system to favor simpler, more elegant formulas (a form of Occam's Razor) and avoid overfitting.

### 3. Selection
The "fittest" equations—those with the best combination of accuracy and simplicity—are selected to become the "parents" for the next generation. This is done using a method called **Tournament Selection**, which favors better solutions while still allowing for some diversity.

### 4. Reproduction
A new generation of "child" equations is created from the selected parents using two main genetic operators:
- **Crossover:** This mimics sexual reproduction by taking two parent equations and swapping parts of their formulas to create two new offspring that combine their parents' traits.
- **Mutation:** This introduces small, random changes into a child equation—like changing a `+` to a `*`—which adds new genetic material to the population and helps the search avoid getting stuck.

This cycle of evaluation, selection, and reproduction repeats for many generations, with each new population of equations becoming progressively better at explaining the data.

---
## The Search Space and Its Constraints

The "search space" is the universe of all possible equations the theorist can build. This space is defined by a set of building blocks, and the search is guided by a set of constraints to ensure the results are useful.

### The Search Space
The theorist constructs equations using the following components:
- **Unary Operators (take one input):** Logarithm (`log`), Exponent (`exp`).
- **Binary Operators (take two inputs):** Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`), Power (`^`).
- **Terminals:** These are the endpoints of the equation, which include the variables from the dataset (e.g., `n_digits`) and a set of tunable constants (e.g., `c1`, `c2`).

### How the Search is Constrained
To find good solutions efficiently, the search is guided by several important constraints:
- **Complexity Limit:** There is a hard limit on the size of any equation. Any formula that grows to have more than 40 symbols is immediately discarded. This prevents the system from exploring extremely long and uninterpretable equations.
- **Complexity Penalty:** In addition to the hard limit, there is a soft penalty for complexity. During the evaluation step, longer equations receive a worse fitness score. This creates an evolutionary pressure that favors shorter and simpler formulas that are more likely to be generalizable (a form of Occam's Razor).
- **Mathematical Validity:** The system automatically discards any equation that produces a mathematical error (like division by zero or the logarithm of a negative number) during its evaluation. This ensures that only valid formulas survive and reproduce.