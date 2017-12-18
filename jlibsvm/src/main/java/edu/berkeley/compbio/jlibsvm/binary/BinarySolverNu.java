package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.Solver_NU;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BinarySolverNu<L extends Comparable> extends Solver_NU<L> {

// --------------------------- CONSTRUCTORS ---------------------------

  public BinarySolverNu(
      List<SolutionVector> solutionVectors,
      QMatrix Q,
      double Cp,
      double Cn,
      double eps,
      boolean shrinking,
      int maxIterations) {
    super(solutionVectors, Q, Cp, Cn, eps, shrinking, maxIterations);
  }

// -------------------------- OTHER METHODS --------------------------

  public BinaryModel<L> solve() {
    int iter = optimize();

    BinaryModel<L> model = new BinaryModel<>();

    // calculate rho

    calculate_rho(model);

    // calculate objective value

    double v = 0;
    for (SolutionVector svC : active) {
      v += svC.alpha * (svC.G + svC.linearTerm);
    }

    model.obj = v / 2;

    // put the solution, mapping the alphas back to their original order

    model.supportVectors = new HashMap<SparseVector, Double>();
    for (SolutionVector svC : allExamples) {
      model.supportVectors.put(svC.point, svC.alpha);
    }

    // note at this point the solution includes _all_ vectors, even if their alphas are zero

    // we can't do this yet because in the regression case there are twice as many alphas as vectors
    // model.compact();

    model.upperBoundPositive = Cp;
    model.upperBoundNegative = Cn;

    Logger.getGlobal().info("optimization finished, #iter = " + iter);

    return model;
  }
}
