package edu.berkeley.compbio.jlibsvm;

import com.google.common.collect.Multiset;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * An SVM problem consisting of a mapping of training examples to labels.
 * <p/>
 * The generic parameters are L, the label type; P, the type of objects to be classified, and R, the concrete type of the problem itself.
 * <p/>
 * An SVM problem
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public interface SvmProblem<L extends Comparable, R> {
// -------------------------- OTHER METHODS --------------------------

  Multiset<L> getExampleCounts();

  Map<SparseVector, L> getExamples();

  List<L> getLabels();

  int getNumExamples();

  ScalingModel getScalingModel();

  L getTargetValue(SparseVector point);

  Stream<R> makeFolds(int numberOfFolds);

  Set<SparseVector> getHeldOutPoints();
}
