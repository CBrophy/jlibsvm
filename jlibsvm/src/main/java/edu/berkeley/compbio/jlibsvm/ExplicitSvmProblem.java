package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.List;
import java.util.Map;

/**
 * This may seem pointless, but it helps with the generics spaghetti by constraining the type R.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
@Deprecated
public interface ExplicitSvmProblem<L extends Comparable, R extends SvmProblem<L, R>> extends
    SvmProblem<L, R> {
// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface SvmProblem ---------------------

  Map<SparseVector, L> getExamples();

  List<L> getLabels();

  L getTargetValue(SparseVector point);

// -------------------------- OTHER METHODS --------------------------


}
