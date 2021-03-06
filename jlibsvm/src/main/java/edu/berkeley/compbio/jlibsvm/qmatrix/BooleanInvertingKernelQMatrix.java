package edu.berkeley.compbio.jlibsvm.qmatrix;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BooleanInvertingKernelQMatrix extends BasicKernelQMatrix {
// --------------------------- CONSTRUCTORS ---------------------------

  public BooleanInvertingKernelQMatrix( KernelFunction kernel, int numExamples,
      int maxCachedRank) {
    super(kernel, numExamples, maxCachedRank);
  }

// -------------------------- OTHER METHODS --------------------------

  public double computeQ(SolutionVector a, SolutionVector b) {
    return (((a.targetValue == b.targetValue) ? 1 : -1) * kernel
        .evaluate(a.point, b.point));
  }
}
