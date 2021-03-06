package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SolutionVector;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.qmatrix.BooleanInvertingKernelQMatrix;
import edu.berkeley.compbio.jlibsvm.qmatrix.QMatrix;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class C_SVC<L extends Comparable> extends BinaryClassificationSVM<L> {
// -------------------------- OTHER METHODS --------------------------

  @Override
  public BinaryModel<L> trainOne(BinaryClassificationProblem<L> problem, double Cp, double Cn,
       ImmutableSvmParameterPoint<L> param) {
    double linearTerm = -1.0;
    Map<SparseVector, Boolean> examples = problem.getBooleanExamples();

    List<SolutionVector> solutionVectors = new ArrayList<>(examples.size());

    for (Map.Entry<SparseVector, Boolean> example : examples.entrySet()) {
      SolutionVector sv =
          new SolutionVector(example.getKey().getId(), example.getKey(),
              example.getValue(),
              linearTerm);
      solutionVectors.add(sv);
    }

    QMatrix qMatrix =
        new BooleanInvertingKernelQMatrix(param.kernel, solutionVectors.size(),
            param.getCacheRows());

    BinarySolver<L> s = new BinarySolver<L>(
        solutionVectors,
        qMatrix,
        Cp,
        Cn,
        param.eps,
        param.shrinking,
        param.maxIterations);

    BinaryModel<L> model = s.solve();

    model.param = param;
    model.trueLabel = problem.getTrueLabel();
    model.falseLabel = problem.getFalseLabel();
    model.setScalingModel(problem.getScalingModel());

    if (Cp == Cn) {
      Logger.getGlobal().info("nu = " + model.getSumAlpha() / (Cp * problem.getNumExamples()));
    }

    for (Map.Entry<SparseVector, Double> entry : model.supportVectors.entrySet()) {
      final SparseVector key = entry.getKey();
      final Boolean target = examples.get(key);
      if (!target) {
        entry.setValue(entry.getValue() * -1);
      }
    }

    model.compact();

    return model;
  }

  public String getSvmType() {
    return "c_svc";
  }

  @Override
  public void validateParam( ImmutableSvmParameter<L> param) {
    super.validateParam(param);
    if (param instanceof ImmutableSvmParameterPoint) {
      if (((ImmutableSvmParameterPoint) param).C <= 0) {
        throw new SvmException("C <= 0");
      }
    }
  }
}
