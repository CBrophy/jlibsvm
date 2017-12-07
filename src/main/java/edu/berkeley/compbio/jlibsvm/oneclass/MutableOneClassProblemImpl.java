package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableOneClassProblemImpl<L, P extends SparseVector> extends OneClassProblemImpl<L, P>
    implements MutableSvmProblem<Double, P, OneClassProblem<L, P>> {
// --------------------------- CONSTRUCTORS ---------------------------

  public MutableOneClassProblemImpl(int numExamples, L label) {
    super(new HashMap<P, Double>(numExamples), label);
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MutableSvmProblem ---------------------

  public void addExample(P point, Double label) {
    examples.put(point, label);
  }

  public void addExampleFloat(P point, Double x) {
    addExample(point, x);
  }
}
