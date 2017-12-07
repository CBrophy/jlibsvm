package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class NoopScalingModelLearner<P extends SparseVector> implements ScalingModelLearner<P> {
// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModelLearner ---------------------

  /**
   * default implementation returns the identity ScalingModel; override for interesting behavior
   */
  public ScalingModel<P> learnScaling(Iterable<P> examples) {
    return new NoopScalingModel<P>();
  }
}
