package edu.berkeley.compbio.jlibsvm.multi;

import edu.berkeley.compbio.jlibsvm.SvmProblem;
import edu.berkeley.compbio.jlibsvm.labelinverter.LabelInverter;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Map;
import java.util.Set;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface MultiClassProblem<L extends Comparable> extends SvmProblem<L, MultiClassProblem<L>> {
// -------------------------- OTHER METHODS --------------------------

  Map<L, Set<SparseVector>> getExamplesByLabel();

  Class getLabelClass();

  LabelInverter<L> getLabelInverter();

  MultiClassProblem<L> getScaledCopy(ScalingModelLearner scalingModelLearner);
}
