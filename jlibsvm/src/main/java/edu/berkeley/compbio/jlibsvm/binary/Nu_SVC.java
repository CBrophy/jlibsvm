package edu.berkeley.compbio.jlibsvm.binary;

import com.google.common.collect.Multiset;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.qmatrix.BooleanInvertingKernelQMatrix;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class Nu_SVC<L extends Comparable> extends BinaryClassificationSVM<L> {


// -------------------------- OTHER METHODS --------------------------

  @Override
  public BinaryModel<L> trainOne(BinaryClassificationProblem<L> problem, double Cp, double Cn,
      @NotNull ImmutableSvmParameterPoint<L> param) {
    if (Cp != 1.0 || Cn != 1.0) {
      Logger.getGlobal().warning("Nu_SVC ignores Cp and Cn, provided values " + Cp + " and " + Cn + " + not used");
    }

    if (!isFeasible(problem, param)) {
      throw new SvmException("Nu_SVM is not feasible for this problem");
    }

    int l = problem.getNumExamples();

    double nu = param.nu;

    double sumPos = nu * l / 2;
    double sumNeg = nu * l / 2;

    Map<SparseVector, Boolean> examples = problem.getBooleanExamples();

    double linearTerm = 0.0;
    List<SolutionVector> solutionVectors = new ArrayList<>();

    for (Map.Entry<SparseVector, Boolean> entry : examples.entrySet()) {
      double initAlpha;
      if (entry.getValue()) {
        initAlpha = Math.min(1.0, sumPos);
        sumPos -= initAlpha;
      } else {
        initAlpha = Math.min(1.0, sumNeg);
        sumNeg -= initAlpha;
      }
      SolutionVector sv =
          new SolutionVector(entry.getKey().getId(), entry.getKey(), entry.getValue(),
              linearTerm,
              initAlpha);

      solutionVectors.add(sv);
    }

    QMatrix qMatrix =
        new BooleanInvertingKernelQMatrix(param.kernel, problem.getNumExamples(),
            param.getCacheRows());
    BinarySolverNu<L> s =
        new BinarySolverNu<>(solutionVectors, qMatrix, 1.0, 1.0, param.eps, param.shrinking, param.maxIterations);

    BinaryModel<L> model = s.solve();

    model.param = param;
    model.trueLabel = problem.getTrueLabel();
    model.falseLabel = problem.getFalseLabel();

    double r = model.r;

    Logger.getGlobal().info("C = " + 1 / r);

    for (Map.Entry<SparseVector, Double> entry : model.supportVectors.entrySet()) {
      entry.setValue((examples.get(entry.getKey()) ? 1. : -1.) / r);
    }

    model.rho /= r;
    model.obj /= r * r;
    model.upperBoundPositive = 1 / r;
    model.upperBoundNegative = 1 / r;

    model.compact();

    return model;
  }

  public boolean isFeasible(BinaryClassificationProblem problem,
      @NotNull ImmutableSvmParameter<L> param) {
    Multiset<Boolean> counts = problem.getExampleCounts();

    int n1 = counts.count(problem.getTrueLabel());
    int n2 = counts.count(problem.getFalseLabel());

    if (param.nu * (n1 + n2) / 2 > Math.min(n1, n2)) {
      return false; //"specified nu is infeasible";
    }

    return true;
  }

  public String getSvmType() {
    return "nu_svc";
  }

  public void validateParam(@NotNull ImmutableSvmParameter<L> param) {
    super.validateParam(param);
    if (param.nu <= 0 || param.nu > 1) {
      throw new SvmException("nu <= 0 or nu > 1");
    }
  }
}
