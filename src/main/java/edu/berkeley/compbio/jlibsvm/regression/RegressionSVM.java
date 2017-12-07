package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.SVM;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class RegressionSVM<P extends SparseVector, R extends RegressionProblem<P, R>> extends SVM<Double, P, R> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(RegressionSVM.class);

  private final double SQRT_2 = (double) Math.sqrt(2);

// -------------------------- OTHER METHODS --------------------------

  // Return parameter of a Laplace distribution

  protected double laplaceParameter(RegressionProblem<P, R> problem,
      @NotNull ImmutableSvmParameter<Double, P> param)
  //,   final TreeExecutorService execService)
  {
    int i;
    double mae = 0;

    Map<P, Double> ymv = continuousCrossValidation(problem, param); //, execService);

    for (Map.Entry<P, Double> entry : ymv.entrySet()) {
      double newVal = problem.getTargetValue(entry.getKey()) - entry.getValue();
      entry.setValue(newVal);
      mae += Math.abs(newVal);
    }

    mae /= problem.getNumExamples();

    double std = SQRT_2 * mae;
    int count = 0;
    mae = 0;

    for (Map.Entry<P, Double> entry : ymv.entrySet()) {
      double absVal = Math.abs(entry.getValue());
      if (absVal > 5 * std) {
        count = count + 1;
      } else {
        mae += absVal;
      }
    }
    mae /= (problem.getNumExamples() - count);
    logger.info("Prob. model for test data: target value = predicted value + z");
    logger.info("z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae);
    return mae;
  }

  public abstract RegressionModel<P> train(R problem,
      @NotNull ImmutableSvmParameter<Double, P> param);
  //,final TreeExecutorService execService);

  @Override
  public void validateParam(@NotNull ImmutableSvmParameter<Double, P> param) {
    super.validateParam(param);
  }


  public RegressionCrossValidationResults<P, R> performCrossValidation(R problem,
      @NotNull ImmutableSvmParameter<Double, P> param)
  //,final TreeExecutorService execService)
  {
    Map<P, Double> decisionValues = continuousCrossValidation(problem, param); //, execService);

    RegressionCrossValidationResults<P, R> cv = new RegressionCrossValidationResults<P, R>(problem,
        decisionValues);
    return cv;
  }
}
