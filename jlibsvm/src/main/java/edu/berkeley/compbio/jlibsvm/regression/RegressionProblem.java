package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.SvmProblem;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface RegressionProblem<R> extends SvmProblem<Double, R> {

  R getScaledCopy(ScalingModelLearner scalingModelLearner);
}

