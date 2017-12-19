package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SvmProblem;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface BinaryClassificationProblem<L extends Comparable>
    extends SvmProblem<L, BinaryClassificationProblem<L>> {
// -------------------------- OTHER METHODS --------------------------

  Map<SparseVector, Boolean> getBooleanExamples();

  L getFalseLabel();


  BinaryClassificationProblem<L> getScaledCopy(ScalingModelLearner scalingModelLearner);

  L getTrueLabel();

  void setupLabels();
}
