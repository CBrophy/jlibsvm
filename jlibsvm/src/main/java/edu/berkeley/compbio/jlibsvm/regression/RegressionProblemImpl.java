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

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionProblemImpl<R extends RegressionProblem<R>> extends
    ExplicitSvmProblemImpl<Double, R>
    implements RegressionProblem<R> {
// --------------------------- CONSTRUCTORS ---------------------------

  public RegressionProblemImpl(Map<SparseVector, Double> examples,
      ScalingModel scalingModel) {
    super(examples, scalingModel);
  }

  public RegressionProblemImpl(Map<SparseVector, Double> examples,
      ScalingModel scalingModel,
      Set<SparseVector> heldOutPoints) {
    super(examples, scalingModel, heldOutPoints);
  }

  public RegressionProblemImpl(RegressionProblemImpl<R> backingProblem,
      Set<SparseVector> heldOutPoints) {
    super(new SubtractionMap<>(backingProblem.examples, heldOutPoints),
        backingProblem.scalingModel, heldOutPoints);
  }

  public RegressionProblemImpl(Map<SparseVector, Double> examples) {
    super(examples);
  }

// ------------------------ INTERFACE METHODS ------------------------

  // cache the scaled copy, taking care that the scalingModelLearner is the same one.
  // only bother keeping one (i.e. don't make a map from learners to scaled copies)
  private ScalingModelLearner lastScalingModelLearner = null;
  private R scaledCopy = null;

// --------------------- Interface SvmProblem ---------------------

  public List<Double> getLabels() {
    throw new SvmException("Shouldn't try to get unique target values for a regression problem");
  }

// -------------------------- OTHER METHODS --------------------------

  protected R makeFold(Set<SparseVector> heldOutPoints) {
    return (R) new RegressionProblemImpl(this, heldOutPoints);
  }

  public R getScaledCopy( ScalingModelLearner scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public R createScaledCopy(Map<SparseVector, Double> scaledExamples,
      ScalingModel learnedScalingModel) {
    return (R) new RegressionProblemImpl<R>(scaledExamples,
        learnedScalingModel);
  }
}
