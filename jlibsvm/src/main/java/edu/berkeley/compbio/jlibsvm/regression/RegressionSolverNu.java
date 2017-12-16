package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.Solver_NU;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionSolverNu extends Solver_NU<Double> {

// --------------------------- CONSTRUCTORS ---------------------------

  public RegressionSolverNu(
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
    int iter = optimize();

    RegressionModel model = new RegressionModel();

    // calculate rho

    calculate_rho(model);

    model.supportVectors = new HashMap<>();
    for (SolutionVector svC : allExamples) {      // the examples contain both a true and a false SolutionVector for each P.			// we want the difference of their alphas
      Double alphaDiff = model.supportVectors.get(svC.point);
      if (alphaDiff == null) {
        alphaDiff = 0.;
      }
      alphaDiff += (svC.targetValue ? 1. : -1.) * svC.alpha;

      model.supportVectors.put(svC.point, alphaDiff);
    }



    // note at this point the solution includes _all_ vectors, even if their alphas are zero

    // we can't do this yet because in the regression case there are twice as many alphas as vectors		// model.compact();

    //	model.upperBoundPositive = Cp;		//	model.upperBoundNegative = Cn;

    Logger.getGlobal().info("optimization finished, #iter = " + iter);

    return model;
  }
}
