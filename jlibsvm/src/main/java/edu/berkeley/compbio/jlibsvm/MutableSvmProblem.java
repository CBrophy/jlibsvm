package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * An SVM problem to which training examples may be added.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface MutableSvmProblem<L extends Comparable, R extends SvmProblem<L, R>>
    extends ExplicitSvmProblem<L, R> {
// -------------------------- OTHER METHODS --------------------------

  void addExample(SparseVector point, L label);

  void addExampleFloat(SparseVector point, Double x);
}
