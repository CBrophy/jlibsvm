package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.Solver;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;
import java.util.List;
import org.apache.log4j.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionSolver extends Solver<Double> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(RegressionSolver.class);

// --------------------------- CONSTRUCTORS ---------------------------

  public RegressionSolver(
      List<SolutionVector> solutionVectors,
      QMatrix Q,
      double C,
      double eps,
      boolean shrinking,
      int maxIterations) {
    super(solutionVectors, Q, C, C, eps, shrinking, maxIterations);
  }

// -------------------------- OTHER METHODS --------------------------

  public RegressionModel solve() {
    optimize();

    RegressionModel model = new RegressionModel();

    // calculate rho

    calculate_rho(model);

    // calculate objective value
    model.supportVectors = new HashMap<>();
    for (SolutionVector svC : allExamples) {
      // the examples contain both a true and a false SolutionVector for each P.
      // we want the difference of their alphas

      Double alphaDiff = model.supportVectors.get(svC.point);
      if (alphaDiff == null) {
        alphaDiff = 0.;
      }
      alphaDiff += (svC.targetValue ? 1. : -1.) * svC.alpha;

      model.supportVectors.put(svC.point, alphaDiff);
    }

    return model;
  }
}
