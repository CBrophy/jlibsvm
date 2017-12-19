package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.qmatrix.BooleanInvertingKernelQMatrix;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class EpsilonSVR<R extends RegressionProblem<R>> extends RegressionSVM<R> {
// -------------------------- OTHER METHODS --------------------------


  public RegressionModel train(R problem,  ImmutableSvmParameter<Double> param) {
    validateParam(param);
    RegressionModel result;
    if (param instanceof ImmutableSvmParameterGrid && param.gridsearchBinaryMachinesIndependently) {
      throw new SvmException(
          "Can't do grid search without cross-validation, which is not implemented for regression SVMs.");
    } else {
      result = trainScaled(problem, (ImmutableSvmParameterPoint<Double>) param);
    }
    return result;
  }


  private RegressionModel trainScaled(R problem,
       ImmutableSvmParameterPoint<Double> param) {
    if (param.scalingModelLearner != null && param.scaleBinaryMachinesIndependently) {
      // the examples are copied before scaling, not scaled in place
      // that way we don't need to worry that the same examples are being used in another thread, or scaled differently in different contexts, etc.
      // this may cause memory problems though

      problem = problem.getScaledCopy(param.scalingModelLearner);
    }

    double laplaceParameter = RegressionModel.NO_LAPLACE_PARAMETER;
    if (param.probability) {
      laplaceParameter = laplaceParameter(problem, param);
    }

    List<SolutionVector> solutionVectors = new ArrayList<>();

    for (Map.Entry<SparseVector, Double> example : problem.getExamples().entrySet()) {
      SolutionVector sv;

      sv = new SolutionVector(example.getKey().getId(), example.getKey(), true,
          param.p - example.getValue());

      solutionVectors.add(sv);

      sv = new SolutionVector(-example.getKey().getId(), example.getKey(), false,
          param.p + example.getValue());

      solutionVectors.add(sv);
    }

    QMatrix qMatrix =
        new BooleanInvertingKernelQMatrix(param.kernel, solutionVectors.size(),
            param.getCacheRows());

    RegressionSolver s = new RegressionSolver(
        solutionVectors,
        qMatrix,
        param.C,
        param.eps,
        param.shrinking,
        param.maxIterations);

    RegressionModel model = s.solve();

    model.param = param;
    model.laplaceParameter = laplaceParameter;

    model.compact();

    return model;
  }

  public String getSvmType() {
    return "epsilon_svr";
  }

  @Override
  public void validateParam( ImmutableSvmParameter<Double> param) {
    super.validateParam(param);
    if (param.p < 0) {
      throw new SvmException("p < 0");
    }
    if (param instanceof ImmutableSvmParameterPoint) {
      if (((ImmutableSvmParameterPoint) param).C <= 0) {
        throw new SvmException("C <= 0");
      }
    }
  }
}
