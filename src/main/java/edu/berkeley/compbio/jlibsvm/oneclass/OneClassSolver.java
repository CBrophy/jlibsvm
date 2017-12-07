package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.Solver;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;
import java.util.List;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class OneClassSolver<L> extends Solver<Double> {
// --------------------------- CONSTRUCTORS ---------------------------

  public OneClassSolver(List<SolutionVector> solutionVectors, QMatrix Q, double C, double eps,
      boolean shrinking) {
    super(solutionVectors, Q, C, C, eps, shrinking);
  }

// -------------------------- OTHER METHODS --------------------------

  public OneClassModel<L> solve() {
    optimize();

    OneClassModel<L> model = new OneClassModel<L>();

    calculate_rho(model);

    model.supportVectors = new HashMap<>();
    for (SolutionVector svC : allExamples) {
      model.supportVectors.put(svC.point, svC.alpha);
    }

    // note at this point the solution includes _all_ vectors, even if their alphas are zero

    // we can't do this yet because in the regression case there are twice as many alphas as vectors		// model.compact();

    return model;
  }
}
