package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class CompositeGaussianRBFKernel extends GammaKernel {
// ------------------------------ FIELDS ------------------------------

  KernelFunction underlyingKernel;

// --------------------------- CONSTRUCTORS ---------------------------

  public CompositeGaussianRBFKernel(double gamma, KernelFunction underlyingKernel) {
    super(gamma);
    this.underlyingKernel = underlyingKernel;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface KernelFunction ---------------------

  public double evaluate(SparseVector x, SparseVector y) {
    // we're looking for the square of the distance between x and y in the original space
    // which equals x_square + y_square - 2 * dot(x, y);

    // ** should cache the squares?
    double xSquare = underlyingKernel.evaluate(x, x);
    double ySquare = underlyingKernel.evaluate(y, y);

    double differenceNormSquared = xSquare + ySquare - (2.0 * underlyingKernel.evaluate(x, y));

    double result = Math.exp(-gamma * differenceNormSquared);

    return result;
  }
}
