package edu.berkeley.compbio.jlibsvm.multi;

import edu.berkeley.compbio.jlibsvm.ExplicitSvmProblemImpl;
import edu.berkeley.compbio.jlibsvm.labelinverter.LabelInverter;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.jlibsvm.util.SubtractionMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MultiClassProblemImpl<L extends Comparable>
    extends ExplicitSvmProblemImpl<L, MultiClassProblem<L>>
    implements MultiClassProblem<L> {
// ------------------------------ FIELDS ------------------------------

  Class labelClass;

  private LabelInverter<L> labelInverter;

  // cache the scaled copy, taking care that the scalingModelLearner is the same one.
  // only bother keeping one (i.e. don't make a map from learners to scaled copies)
  private ScalingModelLearner lastScalingModelLearner = null;
  private MultiClassProblem<L> scaledCopy = null;


  private Map<L, Set<SparseVector>> theInverseMap = null;

// --------------------------- CONSTRUCTORS ---------------------------

  /**
   * For now, pending further cleanup, we need to create arrays of the label type.  That's impossible to do with generics
   * alone, so we need to provide the class object (e.g., String.class or whatever) for the label type used.  Of course
   * this should match the generics used on SvmProblem, etc.
   */
  public MultiClassProblemImpl(Class labelClass, LabelInverter<L> labelInverter,
      Map<SparseVector, L> examples,
      ScalingModel scalingModel) {
    super(examples, scalingModel);
    this.labelClass = labelClass;
    this.labelInverter = labelInverter;
  }

  public MultiClassProblemImpl(MultiClassProblemImpl<L> backingProblem,
      Set<SparseVector> heldOutPoints) {
    super(new SubtractionMap(backingProblem.examples, heldOutPoints),
        backingProblem.scalingModel, heldOutPoints);
    this.labelClass = backingProblem.labelClass;
    this.labelInverter = backingProblem.labelInverter;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public Class getLabelClass() {
    return labelClass;
  }

  public LabelInverter<L> getLabelInverter() {
    return labelInverter;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MultiClassProblem ---------------------

  public Map<L, Set<SparseVector>> getExamplesByLabel() {
    if (theInverseMap == null) {
      theInverseMap = new HashMap<>();
      for (L label : getLabels()) {
        theInverseMap.put(label, new HashSet<>());
      }

      // separate the training set into label-specific sets, caching all the while
      // (too bad the svm training requires all examples in memory)

      // The Apache or Google collections should have a reversible map...?  Whatever, do it by hand

      for (Map.Entry<SparseVector, L> entry : examples.entrySet()) {
        theInverseMap.get(entry.getValue()).add(entry.getKey());
        //examples.put(label, sample);
      }
    }
    return theInverseMap;
  }


  public MultiClassProblem<L> getScaledCopy(
       ScalingModelLearner scalingModelLearner) {
    if (!scalingModelLearner.equals(lastScalingModelLearner)) {
      scaledCopy = learnScaling(scalingModelLearner);
      lastScalingModelLearner = scalingModelLearner;
    }
    return scaledCopy;
  }

  public MultiClassProblem<L> createScaledCopy(Map<SparseVector, L> scaledExamples,
      ScalingModel learnedScalingModel) {
    return new MultiClassProblemImpl<L>(labelClass, labelInverter, scaledExamples,
        learnedScalingModel);
  }

// -------------------------- OTHER METHODS --------------------------

  protected MultiClassProblem<L> makeFold(Set<SparseVector> heldOutPoints) {
    return new MultiClassProblemImpl(this, heldOutPoints);
  }
}
