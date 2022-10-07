# CM-project
This repo implements 2 algorithms, the gradient descent and the conjugate gradient in order to solve the problem of estimating the  2-norm matrix as an uncostrained optimization problem.

# Prerequisites
The project is written in python so in order to install all the dependencies just run the following command on your shell:
``` bash
pip install -r requirements.txt
```

# How to use
## To check our results
You have several experiments performed on the 5 matrices we generated as specified in the report:
* One using Gradient Descent (GD.ipynb)
* Two using Conjugate Gradient Descent once with FR and once with PR to calculate beta (CG-FR.ipynb & CG-PR.ipynb)
* One that graphically compares Gradient Descent and Conjugate Gradient Descent (algorithm-comparisons.ipynb)


## From Scratch
In order to test the algorithms you can create a jupyter notebook or a python script, just use the matrices already generated or generate new matrices running the script "matrix_generator.py", then import the classes that implements the two algorithms (gradient descent and conjugate gradient) and the norm function and load the matrix as specified below:
``` python
import src.normFunc as nf
import src.gd as GD
import src.cg as CG
M1 = np.loadtxt(matrix_path)
```
To initialize the norm function on the matrix, generate the starting point and choose the algorithm which has to be tested, futhermore if you are interestend in the convergence rate just solve the problem with an off-shell algorithm and pass the solution to the algorithm. Finally run the algorithm with the function "run" passing the number of iterations:
``` python
from src.utility import generate_starting_point
f = nf.Norm(M1)
x0 = generate_starting_point(f.dim)
SGD = GD.gradientDescent(f, x0, eps = 1e-5, fstar norm, verbose = True)
gradient, values, errors = SGD.run(500)
```
To make a plot use the function make_plot defined in the utility script
``` python
from src.utility import make_plot
make_plot(gradient, errors, plot_path='plot-gd/', type = 'M1')
```
