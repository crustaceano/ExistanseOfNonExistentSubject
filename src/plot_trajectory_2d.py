import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_levels(func, xrange=None, yrange=None, levels=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-5, 5]
    if levels is None:
        levels = [0, 0.25, 1, 4, 9, 16, 25]
        
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    CS = plt.contour(X, Y, Z, levels=levels, colors='k', linewidths=2.0)
    plt.clabel(CS, inline=1, fontsize=8) 
    plt.grid()              

        
def plot_trajectory(func, history, fit_axis=False, label=None, log=None):
    """
    Plotting the trajectory of a method. 
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    plt.plot(x_values, y_values, '-v', linewidth=5.0, ms=12.0, 
             alpha=1.0, c='r', label=label)
    if log == True:
        plt.xscale('log')
        plt.yscale('log')
    
    # Tries to adapt axis-ranges for the trajectory:
    
    if fit_axis:
        COEF = 1.5
        if log:
            x_positive = [x for x in x_values if x > 0]
            y_positive = [y for y in y_values if y > 0]
            if x_positive and y_positive:
                plt.xlim(min(x_positive) / COEF, max(x_positive) * COEF)
                plt.ylim(min(y_positive) / COEF, max(y_positive) * COEF)
        else:
            xmax, ymax = np.max(np.abs(x_values)), np.max(np.abs(y_values))
            xrange = [-xmax * COEF, xmax * COEF]
            yrange = [-ymax * COEF, ymax * COEF]
            plt.xlim(xrange)
            plt.ylim(yrange)

