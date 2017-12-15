package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * An SVM solution which can classify unknown points using a set of discrete labels.
 *
 * The generic parameters are L: the label type, and P, the type of the objects to be classified.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface DiscreteModel<L> {
// -------------------------- OTHER METHODS --------------------------

  L predictLabel(SparseVector x);
}
