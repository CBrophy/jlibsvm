package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * An SVM solution which assigns a floating-point value to unknown points.
 *
 * The generic parameters is P: the type of the objects to be classified.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface ContinuousModel {
// -------------------------- OTHER METHODS --------------------------

  double predictValue(SparseVector x);
}
