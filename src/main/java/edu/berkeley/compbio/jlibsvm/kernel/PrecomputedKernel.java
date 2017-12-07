package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Properties;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class PrecomputedKernel implements KernelFunction {
// --------------------------- CONSTRUCTORS ---------------------------

  public PrecomputedKernel() {
    throw new UnsupportedOperationException();
  }

  public PrecomputedKernel(Properties props) {
    throw new UnsupportedOperationException();// ** Hmm, not sure how to load precomputed kernels;
  }

// ------------------------ CANONICAL METHODS ------------------------

  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("kernel_type precomputed\n");
    return sb.toString();
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface KernelFunction ---------------------

  public double evaluate(SparseVector x, SparseVector y) {
    return evaluateF(x, y);
  }

// -------------------------- OTHER METHODS --------------------------

  public double evaluateF(SparseVector x, SparseVector y) {
    return x.getValues()[(int) (y.getValues()[0])];
  }
}
