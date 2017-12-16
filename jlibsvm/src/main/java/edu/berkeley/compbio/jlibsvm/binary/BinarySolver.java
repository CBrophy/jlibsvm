package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.Solver;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import java.util.HashMap;
import java.util.List;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BinarySolver<L extends Comparable> extends Solver<L> {
// --------------------------- CONSTRUCTORS ---------------------------

  public BinarySolver(
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
    optimize();

    BinaryModel<L> model = new BinaryModel<>();

    // calculate rho
    calculate_rho(model);

    // calculate objective value

    double v = 0;
    for (SolutionVector svC : allExamples) {
      v += svC.alpha * (svC.G + svC.linearTerm);
    }

    model.obj = v / 2;

    model.supportVectors = new HashMap<>();
    for (SolutionVector svC : allExamples) {
      model.supportVectors.put(svC.point, svC.alpha);
    }

    // note at this point the solution includes _all_ vectors, even if their alphas are zero

    // we can't do this yet because in the regression case there are twice as many alphas as vectors
    // model.compact();

    model.upperBoundPositive = Cp;
    model.upperBoundNegative = Cn;

    return model;
  }
}
