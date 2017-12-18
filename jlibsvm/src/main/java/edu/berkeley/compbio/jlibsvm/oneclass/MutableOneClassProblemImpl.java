package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableOneClassProblemImpl<L> extends OneClassProblemImpl<L>
    implements MutableSvmProblem<Double, OneClassProblem<L>> {
// --------------------------- CONSTRUCTORS ---------------------------

  public MutableOneClassProblemImpl(int numExamples, L label) {
    super(new HashMap<>(numExamples), label);
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MutableSvmProblem ---------------------

  public void addExample(SparseVector point, Double label) {
    examples.put(point, label);
  }

  public void addExampleFloat(SparseVector point, Double x) {
    addExample(point, x);
  }
}
