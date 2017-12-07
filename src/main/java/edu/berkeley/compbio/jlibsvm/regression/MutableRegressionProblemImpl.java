package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableRegressionProblemImpl<P extends SparseVector> extends
    RegressionProblemImpl<P, MutableRegressionProblemImpl<P>>
    implements MutableSvmProblem<Double, P, MutableRegressionProblemImpl<P>> {
// --------------------------- CONSTRUCTORS ---------------------------

  public MutableRegressionProblemImpl(int numExamples) {
    super(new HashMap<P, Double>(numExamples));
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
