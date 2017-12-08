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
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class C_SVC<L extends Comparable> extends BinaryClassificationSVM<L> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(C_SVC.class);

// -------------------------- OTHER METHODS --------------------------

  @Override
  public BinaryModel<L> trainOne(BinaryClassificationProblem<L> problem, double Cp, double Cn,
      @NotNull ImmutableSvmParameterPoint<L> param) {
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
    model.setSvmType(getSvmType());
    model.setScalingModel(problem.getScalingModel());

    if (Cp == Cn) {
      logger.debug("nu = " + model.getSumAlpha() / (Cp * problem.getNumExamples()));
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
  public void validateParam(@NotNull ImmutableSvmParameter<L> param) {
    super.validateParam(param);
    if (param instanceof ImmutableSvmParameterPoint) {
      if (((ImmutableSvmParameterPoint) param).C <= 0) {
        throw new SvmException("C <= 0");
      }
    }
  }
}
