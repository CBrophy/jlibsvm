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
public interface MultiClassProblem<L extends Comparable, P extends SparseVector
    > extends SvmProblem<L, P, MultiClassProblem<L, P>> {
// -------------------------- OTHER METHODS --------------------------

  Map<L, Set<P>> getExamplesByLabel();

  Class getLabelClass();

  LabelInverter<L> getLabelInverter();

  MultiClassProblem<L, P> getScaledCopy(ScalingModelLearner<P> scalingModelLearner);
}
