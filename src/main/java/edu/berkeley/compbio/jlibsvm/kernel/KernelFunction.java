package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface KernelFunction {
// -------------------------- OTHER METHODS --------------------------

  double evaluate(SparseVector x, SparseVector y);

}
