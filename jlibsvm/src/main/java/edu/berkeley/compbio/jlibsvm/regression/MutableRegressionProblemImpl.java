package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableRegressionProblemImpl extends
    RegressionProblemImpl<MutableRegressionProblemImpl>
    implements MutableSvmProblem<Double, MutableRegressionProblemImpl> {
// --------------------------- CONSTRUCTORS ---------------------------

  public MutableRegressionProblemImpl(int numExamples) {
    super(new HashMap<>(numExamples));
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
