package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class NoopScalingModel<P extends SparseVector> implements ScalingModel<P> {
// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModel ---------------------

  /**
   * default implementation just returns the original object, but an overriding implementation that actually changes
   * something should return a copy
   */
  public P scaledCopy(P example) {
    return example;
  }
}
