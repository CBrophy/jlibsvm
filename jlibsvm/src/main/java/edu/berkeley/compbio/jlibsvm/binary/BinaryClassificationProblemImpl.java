package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.ExplicitSvmProblemImpl;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.jlibsvm.util.SubtractionMap;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BinaryClassificationProblemImpl<L extends Comparable>
    extends ExplicitSvmProblemImpl<L, BinaryClassificationProblem<L>>
    implements BinaryClassificationProblem<L>, Serializable {
// ------------------------------ FIELDS ------------------------------

  Class labelClass;

  // these are redundant with getLabels() but clearer which is which
  L trueLabel;
  L falseLabel;

  // cache the scaled copy, taking care that the scalingModelLearner is the same one.
  // only bother keeping one (i.e. don't make a map from learners to scaled copies)
  private ScalingModelLearner lastScalingModelLearner = null;
  private BinaryClassificationProblem<L> scaledCopy = null;

// --------------------------- CONSTRUCTORS ---------------------------

  public BinaryClassificationProblemImpl(Class labelClass, Map<SparseVector, L> examples) {
    super(examples);
    setupLabels();
    this.labelClass = labelClass;
  }

  public void setupLabels() {
    // note the labels are sorted
    List<L> result = super.getLabels();
    if (result != null) {
      falseLabel = result.get(1);
      trueLabel = result.get(0);
    }
  }

  public BinaryClassificationProblemImpl(Class labelClass, Map<SparseVector, L> examples,
       ScalingModel scalingModel, L trueLabel, L falseLabel) {
    super(examples, scalingModel);
    this.trueLabel = trueLabel;
    this.falseLabel = falseLabel;
    this.labelClass = labelClass;
  }

  public BinaryClassificationProblemImpl(BinaryClassificationProblemImpl<L> backingProblem,
      Set<SparseVector> heldOutPoints) {
    super(new SubtractionMap<>(backingProblem.examples, heldOutPoints),
        backingProblem.scalingModel, heldOutPoints);
    this.trueLabel = backingProblem.trueLabel;
    this.falseLabel = backingProblem.falseLabel;
    this.labelClass = backingProblem.labelClass;
  }
// --------------------- GETTER / SETTER METHODS ---------------------

  public L getFalseLabel() {
    return falseLabel;
  }

  public L getTrueLabel() {
    return trueLabel;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassificationProblem ---------------------

  public Map<SparseVector, Boolean> getBooleanExamples() {
    if (labelClass.equals(Boolean.class)) {
      return (Map<SparseVector, Boolean>) examples;
    }

    setupLabels();

    Map<SparseVector, Boolean> result = new HashMap<>(examples.size());
    for (Map.Entry<SparseVector, L> entry : examples.entrySet()) {
      result.put(entry.getKey(), entry.getValue().equals(trueLabel) ? Boolean.TRUE : Boolean.FALSE);
    }
    return result;
  }

  public BinaryClassificationProblem<L> getScaledCopy(
       ScalingModelLearner scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public BinaryClassificationProblem<L> createScaledCopy(Map<SparseVector, L> scaledExamples,
      ScalingModel learnedScalingModel) {
    return new BinaryClassificationProblemImpl<>(labelClass, scaledExamples,
        learnedScalingModel, trueLabel, falseLabel);
  }

// --------------------- Interface SvmProblem ---------------------

  public L getTargetValue(SparseVector point) {
    return examples.get(point);
  }

// -------------------------- OTHER METHODS --------------------------

  protected BinaryClassificationProblem<L> makeFold(Set<SparseVector> heldOutPoints) {
    return new BinaryClassificationProblemImpl(this, heldOutPoints);
  }
// -------------------------- INNER CLASSES --------------------------
}
