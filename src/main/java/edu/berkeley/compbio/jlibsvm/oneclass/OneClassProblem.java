package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.regression.RegressionProblem;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface OneClassProblem<L> extends RegressionProblem<OneClassProblem<L>> {
// -------------------------- OTHER METHODS --------------------------

  L getLabel();

  OneClassProblem<L> getScaledCopy(ScalingModelLearner scalingModelLearner);
}
