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
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class Nu_SVR<R extends RegressionProblem<R>> extends RegressionSVM<R> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(Nu_SVR.class);

// -------------------------- OTHER METHODS --------------------------

  public RegressionModel train(R problem, @NotNull ImmutableSvmParameter<Double> param)
  //,final TreeExecutorService execService)
  {
    validateParam(param);
    RegressionModel result;
    if (param instanceof ImmutableSvmParameterGrid && param.gridsearchBinaryMachinesIndependently) {
      throw new SvmException(
          "Can't do grid search without cross-validation, which is not implemented for regression SVMs.");
    } else {
      result = trainScaled(problem, (ImmutableSvmParameterPoint<Double>) param);//, execService);
    }
    return result;
  }


  private RegressionModel trainScaled(R problem,
      @NotNull ImmutableSvmParameterPoint<Double> param)
  //, final TreeExecutorService execService)
  {
    if (param.scalingModelLearner != null && param.scaleBinaryMachinesIndependently) {
      // the examples are copied before scaling, not scaled in place
      // that way we don't need to worry that the same examples are being used in another thread, or scaled differently in different contexts, etc.
      // this may cause memory problems though

      problem = problem.getScaledCopy(param.scalingModelLearner);
    }

    double laplaceParameter = RegressionModel.NO_LAPLACE_PARAMETER;
    if (param.probability) {
      laplaceParameter = laplaceParameter(problem, param);//, execService);
    }

    double sum = param.C * param.nu * problem.getNumExamples() / 2f;

    List<SolutionVector> solutionVectors = new ArrayList<>();

    for (Map.Entry<SparseVector, Double> example : problem.getExamples().entrySet()) {
      double initAlpha = Math.min(sum, param.C);
      sum -= initAlpha;

      SolutionVector sv;

      sv = new SolutionVector(example.getKey().getId(), example.getKey(), true,
          -example.getValue(),
          initAlpha);
      solutionVectors.add(sv);

      sv = new SolutionVector(-example.getKey().getId(), example.getKey(), false,
          example.getValue(),
          initAlpha);
      solutionVectors.add(sv);
    }

    QMatrix qMatrix =
        new BooleanInvertingKernelQMatrix(param.kernel, solutionVectors.size(),
            param.getCacheRows());
    RegressionSolverNu s =
        new RegressionSolverNu(solutionVectors, qMatrix, param.C, param.eps, param.shrinking);

    RegressionModel model = s.solve();

    model.param = param;
    model.setSvmType(getSvmType());
    model.laplaceParameter = laplaceParameter;

    logger.info("epsilon = " + (-model.r));

    model.compact();

    return model;
  }

  public String getSvmType() {
    return "nu_svr";
  }

  public void validateParam(@NotNull ImmutableSvmParameterPoint<Double> param) {
    super.validateParam(param);
    if (param.nu <= 0 || param.nu > 1) {
      throw new SvmException("nu <= 0 or nu > 1");
    }
    if (param.C <= 0) {
      throw new SvmException("C <= 0");
    }
  }
}
