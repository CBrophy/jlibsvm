package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface ScalingModelLearner {
// -------------------------- OTHER METHODS --------------------------

  public ScalingModel learnScaling(Iterable<SparseVector> examples);
}
