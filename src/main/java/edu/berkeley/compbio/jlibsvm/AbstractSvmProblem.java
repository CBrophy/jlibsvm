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
public abstract class AbstractSvmProblem<L extends Comparable, P extends SparseVector, R extends SvmProblem<L, P, R>>
    implements SvmProblem<L, P, R> {
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


  protected R learnScaling(ScalingModelLearner<P> scalingModelLearner) {
    Map<P, L> examples = getExamples();

    ScalingModel<P> learnedScalingModel = scalingModelLearner.learnScaling(examples.keySet());

    Map<P, L> unscaledExamples = getExamples();
    Map<P, L> scaledExamples = new HashMap<P, L>(examples.size());

    for (Map.Entry<P, L> entry : unscaledExamples.entrySet()) {
      P scaledPoint = learnedScalingModel.scaledCopy(entry.getKey());
      scaledExamples.put(scaledPoint, entry.getValue());
    }

    return createScaledCopy(scaledExamples, learnedScalingModel);
  }

  public abstract R createScaledCopy(Map<P, L> scaledExamples,
      ScalingModel<P> learnedScalingModel);
}
