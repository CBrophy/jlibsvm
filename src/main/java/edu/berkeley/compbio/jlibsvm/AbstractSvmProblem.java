package edu.berkeley.compbio.jlibsvm;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class AbstractSvmProblem<L extends Comparable, R extends SvmProblem<L, R>>
    implements SvmProblem<L, R> {
// ------------------------------ FIELDS ------------------------------

  protected Multiset<L> exampleCounts = null;

// --------------------- GETTER / SETTER METHODS ---------------------

  public Multiset<L> getExampleCounts() {
    if (exampleCounts == null) {
      exampleCounts = HashMultiset.create();
      exampleCounts.addAll(getExamples().values());
    }
    return exampleCounts;
  }


  protected R learnScaling(ScalingModelLearner scalingModelLearner) {
    Map<SparseVector, L> examples = getExamples();

    ScalingModel learnedScalingModel = scalingModelLearner.learnScaling(examples.keySet());

    Map<SparseVector, L> unscaledExamples = getExamples();
    Map<SparseVector, L> scaledExamples = new HashMap<SparseVector, L>(examples.size());

    for (Map.Entry<SparseVector, L> entry : unscaledExamples.entrySet()) {
      SparseVector scaledPoint = learnedScalingModel.scaledCopy(entry.getKey());
      scaledExamples.put(scaledPoint, entry.getValue());
    }

    return createScaledCopy(scaledExamples, learnedScalingModel);
  }

  public abstract R createScaledCopy(Map<SparseVector, L> scaledExamples,
      ScalingModel learnedScalingModel);
}
