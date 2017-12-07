package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.ml.CrossValidationResults;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionCrossValidationResults<R extends RegressionProblem<R>> extends
    CrossValidationResults {

  double meanSquaredError;
  double squaredCorrCoeff;

  public RegressionCrossValidationResults(RegressionProblem<R> problem,
      Map<SparseVector, Double> decisionValues) {
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double total_error = 0;
    int numExamples = problem.getNumExamples();

    for (Map.Entry<SparseVector, Double> entry : problem.getExamples().entrySet()) {
      SparseVector p = entry.getKey();
      Double y = entry.getValue();
      Double v = decisionValues.get(p);
      total_error += (v - y) * (v - y);
      sumv += v;
      sumy += y;
      sumvv += v * v;
      sumyy += y * y;
      sumvy += v * y;
    }
    meanSquaredError = total_error / numExamples;
    squaredCorrCoeff =
        ((numExamples * sumvy - sumv * sumy) * (numExamples * sumvy - sumv * sumy)) / (
            (numExamples * sumvv - sumv * sumv) * (numExamples * sumyy - sumy * sumy));

    System.out.print("Cross Validation Mean squared error = " + meanSquaredError + "\n");

    System.out
        .print("Cross Validation Squared correlation coefficient = " + squaredCorrCoeff + "\n");
  }

  public float accuracy() {
    return 0;
  }

  public float accuracyGivenClassified() {
    return 0;
  }

  public float unknown() {
    return 0;
  }
}
