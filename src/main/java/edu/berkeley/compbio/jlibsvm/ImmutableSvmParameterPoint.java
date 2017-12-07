package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ImmutableSvmParameterPoint<L extends Comparable, P extends SparseVector> extends
    ImmutableSvmParameter<L, P> {

  public final double C;// for C_SVC, EPSILON_SVR and NU_SVR
  public final KernelFunction<P> kernel;

  public ImmutableSvmParameterPoint(Builder<L, P> copyFrom) {
    super(copyFrom);
    C = copyFrom.C;

    kernel = copyFrom.kernel;

    if (kernel == null) {
      throw new SvmException("Can't build a parameter set with no kernel");
    }
  }


	/*public static <L extends Comparable, P> Builder<L, P> builder()
		{
		return new Builder<L, P>();
		}
*/

  public static <L extends Comparable, P extends SparseVector> Builder<L, P> asBuilder(
      ImmutableSvmParameter.Builder copyFrom) {
    return new Builder<L, P>(copyFrom);
  }

  public Builder<L, P> asBuilder() {
    return new Builder<L, P>(this);
  }

  public static class Builder<L extends Comparable, P extends SparseVector> extends ImmutableSvmParameter.Builder {

    public double C = 1F;// for C_SVC, EPSILON_SVR and NU_SVR
    public KernelFunction<P> kernel;

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

    public ImmutableSvmParameterPoint<L, P> build() {
      return new ImmutableSvmParameterPoint<L, P>(this);
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
