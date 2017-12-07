package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.regression.RegressionProblemImpl;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class OneClassProblemImpl<L, P extends SparseVector> extends RegressionProblemImpl<P, OneClassProblem<L, P>>
    implements OneClassProblem<L, P> {
// ------------------------------ FIELDS ------------------------------

  L label;

// --------------------------- CONSTRUCTORS ---------------------------

  public OneClassProblemImpl(Map<P, Double> examples,
      L label)  // set<P> examples
  {
    super(examples);
    this.label = label;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public L getLabel() {
    return label;
  }

  private ScalingModelLearner<P> lastScalingModelLearner = null;
  private OneClassProblemImpl<L, P> scaledCopy = null;

// --------------------------- CONSTRUCTORS ---------------------------


  public OneClassProblemImpl(Map<P, Double> examples, L label,
      ScalingModel<P> learnedScalingModel)  // set<P> examples
  {
    super(examples, learnedScalingModel);
    this.label = label;
  }


  public OneClassProblemImpl<L, P> getScaledCopy(
      @NotNull ScalingModelLearner<P> scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = (OneClassProblemImpl<L, P>) learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public OneClassProblemImpl<L, P> createScaledCopy(Map<P, Double> scaledExamples,
      ScalingModel<P> learnedScalingModel) {
    return new OneClassProblemImpl<L, P>(scaledExamples, label,
        learnedScalingModel);
  }
}
