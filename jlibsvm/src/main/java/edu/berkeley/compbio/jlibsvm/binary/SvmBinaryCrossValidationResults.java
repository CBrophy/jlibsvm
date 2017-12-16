package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SigmoidProbabilityModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.ml.BinaryCrossValidationResults;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SvmBinaryCrossValidationResults<L extends Comparable> extends
    BinaryCrossValidationResults {
// ------------------------------ FIELDS ------------------------------
  SigmoidProbabilityModel sigmoid;

  public SvmBinaryCrossValidationResults(BinaryClassificationProblem<L> problem,
      final Map<SparseVector, Double> decisionValues, boolean probability) {
    // convert to arrays

    int totalExamples = decisionValues.size();

    final double[] decisionValueArray = new double[totalExamples];
    final boolean[] labelArray = new boolean[totalExamples];

    Logger.getGlobal().info("Collecting binary cross-validation results for " + totalExamples + " points");

    L trueLabel = problem.getTrueLabel();

    for (Map.Entry<SparseVector, Double> entry : decisionValues.entrySet()) {
      decisionValueArray[numExamples] = entry.getValue();
      labelArray[numExamples] = problem.getTargetValue(entry.getKey()).equals(trueLabel);
      numExamples++;
    }

    // do this here so that we can forget the arrays
    if (probability) {
      sigmoid = new SigmoidProbabilityModel(decisionValueArray, labelArray);
    }

    // while we're at it, since we've done a cross-validation anyway, we may as well report the accuracy.

    //	tt = 0, ff = 0, ft = 0, tf = 0;
    for (int j = 0; j < numExamples; j++) {
      if (decisionValueArray[j] > 0) {
        if (labelArray[j]) {
          tt++;
        } else {
          ft++;
        }
      } else {
        if (labelArray[j]) {
          tf++;
        } else {
          ff++;
        }
      }
    }

  }

  // --------------------- GETTER / SETTER METHODS ---------------------

  public SigmoidProbabilityModel getSigmoid() {
    return sigmoid;
  }

// -------------------------- OTHER METHODS --------------------------
}
