package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Properties;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class GaussianRBFKernel extends GammaKernel<SparseVector> {


// --------------------------- CONSTRUCTORS ---------------------------
  public GaussianRBFKernel(Properties props) {
    this(Double.parseDouble(props.getProperty("gamma")));
  }

  public GaussianRBFKernel(double gamma) {
    super(gamma);
  }

  // ------------------------ CANONICAL METHODS ------------------------

  @Override
  public String toString() {
    return "RBF gamma=" + gamma;
  }

  // BAD file output infrastructure

  public String toFileOutputString() {
    StringBuilder sb = new StringBuilder();
    sb.append("kernel_type rbf\n");
    sb.append("gamma " + gamma + "\n");
    return sb.toString();
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface KernelFunction ---------------------

  public double evaluate(@NotNull SparseVector x, @NotNull SparseVector y) {
    // try doing the internal stuff at double precision

    return Math.exp(-gamma * SparseVector.squareNorm(x, y));

  }
}
