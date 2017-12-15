package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Collection;
import java.util.HashSet;

/**
 * For now this supports sweeping over C and different kernels (which may have e.g. different gamma).
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ImmutableSvmParameterGrid<L extends Comparable> extends
    ImmutableSvmParameter<L> {

  private final Collection<ImmutableSvmParameterPoint<L>> gridParams;

  public ImmutableSvmParameterGrid(Builder<L> copyFrom) {
    super(copyFrom);
    gridParams = copyFrom.gridParams;
  }

  public Collection<ImmutableSvmParameterPoint<L>> getGridParams() {
    return gridParams;
  }

  public static <L extends Comparable> Builder<L> builder() {
    return new Builder<L>();
  }

  public static class Builder<L extends Comparable> extends ImmutableSvmParameter.Builder {

    public Collection<Double> Cset;
    public Collection<KernelFunction> kernelSet;
    private Collection<ImmutableSvmParameterPoint<L>> gridParams;

    public Builder(ImmutableSvmParameter.Builder copyFrom) {
      super(copyFrom);

      //default
      Cset = new HashSet<>();
      Cset.add(1.0);
    }

    public Builder(ImmutableSvmParameterGrid<L> copyFrom) {
      super(copyFrom);

      gridParams = copyFrom.gridParams;
    }

    public Builder() {
      super();

      //default
      Cset = new HashSet<>();
      Cset.add(1.0);
    }

    public ImmutableSvmParameter<L> build() {
      ImmutableSvmParameterPoint.Builder<L> builder = ImmutableSvmParameterPoint.asBuilder(this);

      if (Cset == null || Cset.isEmpty()) {
        throw new SvmException("Can't build a grid with no C values");
      }

      if (kernelSet == null || kernelSet.isEmpty()) {
        throw new SvmException("Can't build a grid with no kernels");
      }

      if (Cset.size() == 1 && kernelSet.size() == 1) {
        builder.C = Cset.iterator().next();
        builder.kernel = kernelSet.iterator().next();
        return builder.build();
      }
      gridParams = new HashSet<>();

      // the C and kernel set here are ignored; we just overwrite them with the grid points

      for (Double gridC : Cset) {
        for (KernelFunction gridKernel : kernelSet) {
          builder.C = gridC;
          builder.kernel = gridKernel;
          builder.gridsearchBinaryMachinesIndependently = false;

          // this copies all the params so we can safely continue modifying the builder
          gridParams.add(builder.build());
        }
      }

      return new ImmutableSvmParameterGrid<>(this);
    }
  }

  public Builder<L> asBuilder() {
    return new Builder<>(this);
  }
}
