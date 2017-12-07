package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface ScalingModel<P extends SparseVector> {
// -------------------------- OTHER METHODS --------------------------

  P scaledCopy(P example);
}
