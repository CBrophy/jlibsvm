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

public interface QMatrix<P extends SparseVector> {
// -------------------------- OTHER METHODS --------------------------

  double evaluateDiagonal(SolutionVector<P> a);

  void getQ(SolutionVector<P> svA, SolutionVector<P>[] active, double[] buf);

  void getQ(SolutionVector<P> svA, SolutionVector<P>[] active, SolutionVector<P>[] inactive,
      double[] buf);

//	void storeRanks(Collection<SolutionVector<P>> allExamples);

  void initRanks(Collection<SolutionVector<P>> allExamples);

  void maintainCache(SolutionVector<P>[] active, SolutionVector<P>[] newlyInactive);

  public String perfString();
}
