package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class GammaKernel<T extends SparseVector> implements KernelFunction<T> {
// ------------------------------ FIELDS ------------------------------

  public double gamma;

// --------------------------- CONSTRUCTORS ---------------------------

  public GammaKernel(double gamma) {
    this.gamma = gamma;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public double getGamma() {
    return gamma;
  }

  public void setGamma(double gamma) {
    this.gamma = gamma;
  }
}
