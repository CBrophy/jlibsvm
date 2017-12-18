package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ImmutableSvmParameterPoint<L extends Comparable> extends
    ImmutableSvmParameter<L> {

  public final double C;// for C_SVC, EPSILON_SVR and NU_SVR
  public final KernelFunction kernel;

  public ImmutableSvmParameterPoint(Builder<L> copyFrom) {
    super(copyFrom);
    C = copyFrom.C;

    kernel = copyFrom.kernel;

    if (kernel == null) {
      throw new SvmException("Can't build a parameter set with no kernel");
    }
  }

  public static <L extends Comparable> Builder<L> asBuilder(
      ImmutableSvmParameter.Builder copyFrom) {
    return new Builder<>(copyFrom);
  }

  public Builder<L> asBuilder() {
    return new Builder<>(this);
  }

  public static class Builder<L extends Comparable> extends ImmutableSvmParameter.Builder {

    public double C = 1F;// for C_SVC, EPSILON_SVR and NU_SVR
    public KernelFunction kernel;

    public Builder(ImmutableSvmParameter.Builder copyFrom) {
      super(copyFrom);
    }

    public Builder(ImmutableSvmParameterPoint copyFrom) {
      super(copyFrom);
      C = copyFrom.C;
      kernel = copyFrom.kernel;
    }

    public Builder() {
      super();
    }

    public ImmutableSvmParameterPoint<L> build() {
      return new ImmutableSvmParameterPoint<>(this);
    }
  }

  @Override
  public String toString() {
    if (gridsearchBinaryMachinesIndependently) {
      return "C, gamma variable";
    } else {
      return "C=" + C + ", kernel=" + kernel;
    }
  }
}
