package edu.berkeley.compbio.jlibsvm.multi;

import edu.berkeley.compbio.jlibsvm.MutableSvmProblem;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.labelinverter.LabelInverter;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.HashMap;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MutableMultiClassProblemImpl<L extends Comparable> extends
    MultiClassProblemImpl<L>
    implements MutableSvmProblem<L, MultiClassProblem<L>> {
// --------------------------- CONSTRUCTORS ---------------------------

  /**
   * For now, pending further cleanup, we need to create arrays of the label type.  That's impossible to do with generics
   * alone, so we need to provide the class object (e.g., String.class or whatever) for the label type used.  Of course
   * this should match the generics used on SvmProblem, etc.
   */
  public MutableMultiClassProblemImpl(Class labelClass, LabelInverter<L> labelInverter,
      int numExamples,
       ScalingModel scalingModel) {
    super(labelClass, labelInverter, new HashMap<>(numExamples),
        scalingModel);
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface MutableSvmProblem ---------------------

  public void addExample(SparseVector point, L label) {
    examples.put(point, label);
  }

  public void addExampleFloat(SparseVector point, Double x) {
    try {
      addExample(point, (L) labelClass.getConstructor(String.class).newInstance(x.toString()));
    } catch (Exception e) {
      throw new SvmException(e);
    }
  }
}
