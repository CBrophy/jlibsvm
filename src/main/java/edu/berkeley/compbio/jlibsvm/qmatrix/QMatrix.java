package edu.berkeley.compbio.jlibsvm.qmatrix;

import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Collection;


/**
 * Kernel evaluation
 * <p/>
 * the static method k_function is for doing single kernel evaluation
 * <p/>
 * the constructor of Kernel prepares to calculate the l*l kernel matrix
 * <p/>
 * the member function get_Q is for getting one column from the Q Matrix
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */

public interface QMatrix {
// -------------------------- OTHER METHODS --------------------------

  double evaluateDiagonal(SolutionVector a);

  void getQ(SolutionVector svA, SolutionVector[] active, double[] buf);

  void getQ(SolutionVector svA, SolutionVector[] active, SolutionVector[] inactive,
      double[] buf);

  void initRanks(Collection<SolutionVector> allExamples);

  void maintainCache(SolutionVector[] active, SolutionVector[] newlyInactive);

  public String perfString();
}
