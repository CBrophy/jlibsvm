package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SvmProblem;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface BinaryClassificationProblem<L extends Comparable, P extends SparseVector>
    extends SvmProblem<L, P, BinaryClassificationProblem<L, P>> {
// -------------------------- OTHER METHODS --------------------------

  Map<P, Boolean> getBooleanExamples();

  L getFalseLabel();

  @NotNull
  BinaryClassificationProblem<L, P> getScaledCopy(ScalingModelLearner<P> scalingModelLearner);

  L getTrueLabel();

  void setupLabels();
}
