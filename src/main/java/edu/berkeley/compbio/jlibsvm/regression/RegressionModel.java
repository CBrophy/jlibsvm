package edu.berkeley.compbio.jlibsvm.regression;

import edu.berkeley.compbio.jlibsvm.ContinuousModel;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.binary.AlphaModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.ml.CrossValidationResults;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class RegressionModel extends AlphaModel<Double> implements ContinuousModel {

  public ImmutableSvmParameterPoint<Double> param;
// ------------------------------ FIELDS ------------------------------

  public static final double NO_LAPLACE_PARAMETER = -1;

  public double laplaceParameter = NO_LAPLACE_PARAMETER;

  public double r;// for Solver_NU.  I wanted to factor this out as SolutionInfoNu, but that was too much hassle

  public Collection<Double> getLabels() {
    return param.getLabels();
  }

  @Override
  public String getKernelName() {
    return param.kernel.toString();
  }


  public RegressionCrossValidationResults crossValidationResults;

  public CrossValidationResults getCrossValidationResults() {
    return crossValidationResults;
  }
// --------------------------- CONSTRUCTORS ---------------------------

  public RegressionModel() {
    super();
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public double getLaplaceParameter() {
    if (laplaceParameter == NO_LAPLACE_PARAMETER) {
      throw new SvmException("Model doesn't contain information for SVR probability inference\n");
    }
    return laplaceParameter;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ContinuousModel ---------------------

  public double predictValue(SparseVector x) {
    double sum = 0;
    for (int i = 0; i < numSVs; i++) {
      sum += alphas[i] * param.kernel.evaluate(x, SVs[i]);
    }
    sum -= rho;
    return sum;
  }

// -------------------------- OTHER METHODS --------------------------

  public boolean supportsLaplace() {
    return laplaceParameter != NO_LAPLACE_PARAMETER;
  }

  public void writeToStream(DataOutputStream fp) throws IOException {
    super.writeToStream(fp);
    fp.writeBytes("probA " + laplaceParameter + "\n");

    //these must come after everything else
    writeSupportVectors(fp);

    fp.close();
  }
}
