package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.SVM;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class RegressionSVM<R extends RegressionProblem<R>> extends SVM<Double, R> {

  private final double SQRT_2 = (double) Math.sqrt(2);

// -------------------------- OTHER METHODS --------------------------

  // Return parameter of a Laplace distribution

  protected double laplaceParameter(RegressionProblem<R> problem,
       ImmutableSvmParameter<Double> param) {
    int i;
    double mae = 0;

    Map<SparseVector, Double> ymv = continuousCrossValidation(problem, param); //, execService);

    for (Map.Entry<SparseVector, Double> entry : ymv.entrySet()) {
      double newVal = problem.getTargetValue(entry.getKey()) - entry.getValue();
      entry.setValue(newVal);
      mae += Math.abs(newVal);
    }

    mae /= problem.getNumExamples();

    double std = SQRT_2 * mae;
    int count = 0;
    mae = 0;

    for (Map.Entry<SparseVector, Double> entry : ymv.entrySet()) {
      double absVal = Math.abs(entry.getValue());
      if (absVal > 5 * std) {
        count = count + 1;
      } else {
        mae += absVal;
      }
    }
    mae /= (problem.getNumExamples() - count);
    Logger.getGlobal().info("Prob. model for test data: target value = predicted value + z");
    Logger.getGlobal().info("z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae);
    return mae;
  }

  public abstract RegressionModel train(R problem,
       ImmutableSvmParameter<Double> param);

  @Override
  public void validateParam( ImmutableSvmParameter<Double> param) {
    super.validateParam(param);
  }


  public RegressionCrossValidationResults<R> performCrossValidation(R problem,
       ImmutableSvmParameter<Double> param) {
    Map<SparseVector, Double> decisionValues = continuousCrossValidation(problem, param);

    RegressionCrossValidationResults<R> cv = new RegressionCrossValidationResults<>(problem,
        decisionValues);
    return cv;
  }
}
