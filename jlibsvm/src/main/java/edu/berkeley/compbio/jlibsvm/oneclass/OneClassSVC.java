package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.qmatrix.BooleanInvertingKernelQMatrix;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.regression.RegressionModel;
import edu.berkeley.compbio.jlibsvm.regression.RegressionSVM;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class OneClassSVC<L extends Comparable> extends RegressionSVM<OneClassProblem<L>> {

// -------------------------- OTHER METHODS --------------------------


  public RegressionModel train(OneClassProblem<L> problem,
       ImmutableSvmParameter<Double> param) {
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


  private RegressionModel trainScaled(OneClassProblem<L> problem,
       ImmutableSvmParameterPoint<Double> param) {
    if (param.scalingModelLearner != null && param.scaleBinaryMachinesIndependently) {
      // the examples are copied before scaling, not scaled in place
      // that way we don't need to worry that the same examples are being used in another thread, or scaled differently in different contexts, etc.
      // this may cause memory problems though

      problem = problem.getScaledCopy(param.scalingModelLearner);
    }

    double remainingAlpha = param.nu * problem.getNumExamples();

    double linearTerm = 0.0;
    List<SolutionVector> solutionVectors = new ArrayList<>();
    int c = 0;
    for (Map.Entry<SparseVector, Double> example : problem.getExamples().entrySet()) {
      double initAlpha = remainingAlpha > 1.0 ? 1.0 : remainingAlpha;
      remainingAlpha -= initAlpha;

      SolutionVector sv;

      sv = new SolutionVector(example.getKey().getId(), example.getKey(), true,
          linearTerm, initAlpha);
      c++;
      solutionVectors.add(sv);
    }

    QMatrix qMatrix =
        new BooleanInvertingKernelQMatrix(param.kernel, solutionVectors.size(),
            param.getCacheRows());
    OneClassSolver<L> s = new OneClassSolver<>(
        solutionVectors,
        qMatrix,
        1.0,
        param.eps,
        param.shrinking,
        param.maxIterations);

    OneClassModel<L> model = s.solve();

    model.param = param;
    model.label = problem.getLabel();
    model.compact();

    return model;
  }

  public String getSvmType() {
    return "one_class_svc";
  }

  public void validateParam( ImmutableSvmParameterPoint<Double> param) {
    super.validateParam(param);

    if (param.C != 1.0) {
      Logger.getGlobal()
          .warning("OneClassSVC ignores param.C, provided value " + param.C + " + not used");
    }
    if (param.probability) {
      throw new SvmException("one-class SVM probability output not supported yet");
    }
    if (param.nu <= 0 || param.nu > 1) {
      throw new SvmException("nu <= 0 or nu > 1");
    }
  }
}
