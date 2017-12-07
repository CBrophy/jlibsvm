package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableBinaryClassificationProblemImpl<L extends Comparable>
    extends BinaryClassificationProblemImpl<L>
    implements MutableSvmProblem<L, BinaryClassificationProblem<L>> {
// --------------------------- CONSTRUCTORS ---------------------------

  public MutableBinaryClassificationProblemImpl(Class labelClass, int numExamples) {
    super(labelClass, new HashMap<>(numExamples));
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MutableSvmProblem ---------------------

  public void addExample(SparseVector point, L label) {
    examples.put(point, label);
  }

  public void addExampleFloat(SparseVector point, Double x) {
    try {
      //** Should we look up a single label object per string instead of constructing a new one every time?  This depends on L.equals() working right, for instance...
      addExample(point, (L) labelClass.getConstructor(String.class).newInstance(x.toString()));
    } catch (Exception e) {
      throw new SvmException(e);
    }
  }
}
