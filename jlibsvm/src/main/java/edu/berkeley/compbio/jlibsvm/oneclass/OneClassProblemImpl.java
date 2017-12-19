package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.regression.RegressionProblemImpl;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class OneClassProblemImpl<L> extends RegressionProblemImpl<OneClassProblem<L>>
    implements OneClassProblem<L> {
// ------------------------------ FIELDS ------------------------------

  L label;

// --------------------------- CONSTRUCTORS ---------------------------

  public OneClassProblemImpl(Map<SparseVector, Double> examples,
      L label)  // set<P> examples
  {
    super(examples);
    this.label = label;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public L getLabel() {
    return label;
  }

  private ScalingModelLearner lastScalingModelLearner = null;
  private OneClassProblemImpl<L> scaledCopy = null;

// --------------------------- CONSTRUCTORS ---------------------------


  public OneClassProblemImpl(Map<SparseVector, Double> examples, L label,
      ScalingModel learnedScalingModel) {
    super(examples, learnedScalingModel);
    this.label = label;
  }


  public OneClassProblemImpl<L> getScaledCopy(
       ScalingModelLearner scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = (OneClassProblemImpl<L>) learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public OneClassProblemImpl<L> createScaledCopy(Map<SparseVector, Double> scaledExamples,
      ScalingModel learnedScalingModel) {
    return new OneClassProblemImpl<>(scaledExamples, label,
        learnedScalingModel);
  }
}
