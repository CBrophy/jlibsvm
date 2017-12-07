package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * An SVM problem to which training examples may be added.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface MutableSvmProblem<L extends Comparable, P extends SparseVector, R extends SvmProblem<L, P, R>>
    extends ExplicitSvmProblem<L, P, R> {
// -------------------------- OTHER METHODS --------------------------

  void addExample(P point, L label);

  void addExampleFloat(P point, Double x);
}
