package edu.berkeley.compbio.jlibsvm.multi;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.ml.MultiClassCrossValidationResults;
import java.util.Map;
import org.apache.log4j.Logger;


/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SvmMultiClassCrossValidationResults<L extends Comparable> extends
    MultiClassCrossValidationResults<L> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(SvmMultiClassCrossValidationResults.class);

  /**
   * if we did a grid search, keep track of which parameter set was used for these results
   */
  public ImmutableSvmParameter<L> param;

// --------------------------- CONSTRUCTORS ---------------------------


  public SvmMultiClassCrossValidationResults(MultiClassProblem<L> problem,
      Map<SparseVector, L> predictions) {
    super();

    for (Map.Entry<SparseVector, L> entry : problem.getExamples().entrySet()) {
      SparseVector point = entry.getKey();
      L realValue = entry.getValue();
      L predictedValue = predictions.get(point);

      // the confusionMatrix should count predictedValue==null (aka unknown) just like any other value

      addSample(realValue, predictedValue);
    }

    sanityCheck();
  }

// -------------------------- OTHER METHODS --------------------------


  public String getInfo() {
    if (param != null) {
      return param.toString();
    }
    return "";
  }
}
