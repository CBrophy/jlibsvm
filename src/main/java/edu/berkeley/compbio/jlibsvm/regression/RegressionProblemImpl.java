package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.ExplicitSvmProblemImpl;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.jlibsvm.util.SubtractionMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionProblemImpl<P extends SparseVector, R extends RegressionProblem<P, R>> extends
    ExplicitSvmProblemImpl<Double, P, R>
    implements RegressionProblem<P, R> {
// --------------------------- CONSTRUCTORS ---------------------------

  public RegressionProblemImpl(Map<P, Double> examples,
      ScalingModel<P> scalingModel) {
    super(examples, scalingModel);
  }

  public RegressionProblemImpl(Map<P, Double> examples,
      ScalingModel<P> scalingModel,
      Set<P> heldOutPoints) {
    super(examples, scalingModel, heldOutPoints);
  }

  public RegressionProblemImpl(RegressionProblemImpl<P, R> backingProblem, Set<P> heldOutPoints) {
    super(new SubtractionMap<P, Double>(backingProblem.examples, heldOutPoints),
        backingProblem.scalingModel, heldOutPoints);
  }

  public RegressionProblemImpl(Map<P, Double> examples) {
    super(examples);
  }

// ------------------------ INTERFACE METHODS ------------------------

  // cache the scaled copy, taking care that the scalingModelLearner is the same one.
  // only bother keeping one (i.e. don't make a map from learners to scaled copies)
  private ScalingModelLearner<P> lastScalingModelLearner = null;
  private R scaledCopy = null;

// --------------------- Interface SvmProblem ---------------------

  public List<Double> getLabels() {
    throw new SvmException("Shouldn't try to get unique target values for a regression problem");
  }

// -------------------------- OTHER METHODS --------------------------

  protected R makeFold(Set<P> heldOutPoints) {
    return (R) new RegressionProblemImpl(this, heldOutPoints);
  }

  public R getScaledCopy(@NotNull ScalingModelLearner<P> scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public R createScaledCopy(Map<P, Double> scaledExamples,
      ScalingModel<P> learnedScalingModel) {
    return (R) new RegressionProblemImpl<P, R>(scaledExamples,
        learnedScalingModel);
  }
}
