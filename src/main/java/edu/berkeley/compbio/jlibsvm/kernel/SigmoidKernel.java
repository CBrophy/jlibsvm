package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Properties;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SigmoidKernel extends GammaKernel {
// ------------------------------ FIELDS ------------------------------

  public double coef0;

// --------------------------- CONSTRUCTORS ---------------------------

  public SigmoidKernel(Properties props) {
    this(Double.parseDouble(props.getProperty("gamma")),
        Double.parseDouble(props.getProperty("coef0")));
  }

  public SigmoidKernel(double gamma, double coef0) {
    super(gamma);
    this.coef0 = coef0;
  }

// ------------------------ CANONICAL METHODS ------------------------

  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("kernel_type sigmoid\n");
    sb.append("gamma " + gamma + "\n");
    sb.append("coef0 " + coef0 + "\n");
    return sb.toString();
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface KernelFunction ---------------------

  public double evaluate(SparseVector x, SparseVector y) {
    return Math.tanh(gamma * SparseVector.dot(x, y) + coef0);
  }
}
