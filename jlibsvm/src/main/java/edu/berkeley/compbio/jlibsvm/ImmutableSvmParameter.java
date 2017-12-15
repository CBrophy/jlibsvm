package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.multi.MultiClassModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Collection;
import java.util.LinkedHashMap;

/**
 * I do not like it
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class ImmutableSvmParameter<L extends Comparable> {
// ------------------------------ FIELDS ------------------------------

  // these are for training only
  public final double cache_size;// in MB
  public final double eps;// stopping criteria
  public final int maxIterations;
  public final double nu;// for NU_SVC, ONE_CLASS, and NU_SVR
  public final double p;// for EPSILON_SVR
  public final boolean shrinking;// use the shrinking heuristics


  public final double oneVsAllThreshold;
  public final MultiClassModel.OneVsAllMode oneVsAllMode;
  public final MultiClassModel.AllVsAllMode allVsAllMode;
  public final double minVoteProportion;
  public final int falseClassSVlimit;

  /**
   * For unbalanced data, redistribute the misclassification cost C according to the numbers of examples in each class,
   * so that each class has the same total misclassification weight assigned to it and the average is param.C
   */
  public final boolean redistributeUnbalancedC;


  public final boolean scaleBinaryMachinesIndependently;
  public final boolean gridsearchBinaryMachinesIndependently;
  public final boolean normalizeL2;

  public final int crossValidationFolds;

  public final ScalingModelLearner scalingModelLearner;

  public final boolean probability;// do probability estimates
  // We need to maintain the labels (the key on this map) in insertion order
  private final LinkedHashMap<L, Double> weights;

// --------------------------- CONSTRUCTORS ---------------------------

  protected ImmutableSvmParameter(Builder<L> copyFrom) {
    cache_size = copyFrom.cache_size;
    eps = copyFrom.eps;
    weights = new LinkedHashMap<>(copyFrom.weights);
    maxIterations = copyFrom.maxIterations;
    nu = copyFrom.nu;
    p = copyFrom.p;
    shrinking = copyFrom.shrinking;
    probability = copyFrom.probability;
    oneVsAllThreshold = copyFrom.oneVsAllThreshold;
    oneVsAllMode = copyFrom.oneVsAllMode;
    allVsAllMode = copyFrom.allVsAllMode;
    minVoteProportion = copyFrom.minVoteProportion;
    falseClassSVlimit = copyFrom.falseClassSVlimit;
    scaleBinaryMachinesIndependently = copyFrom.scaleBinaryMachinesIndependently;
    normalizeL2 = copyFrom.normalizeL2;
    redistributeUnbalancedC = copyFrom.redistributeUnbalancedC;
    gridsearchBinaryMachinesIndependently = copyFrom.gridsearchBinaryMachinesIndependently;

    scalingModelLearner = copyFrom.scalingModelLearner;
    crossValidationFolds = copyFrom.crossValidationFolds;
  }

// -------------------------- OTHER METHODS --------------------------

  public int getCacheRows() {
    // assume the O(n) term is in the noise
    double mb = cache_size;
    double kb = mb * 1024;
    double bytes = kb * 1024;
    double floats = bytes / 4; // double = 4 bytes
    double floatrows = Math.sqrt(floats);
    //Math.sqrt(floats * 2);
    // the sqrt 2 term is because the cache will be symmetric
    // no it won't
    return (int) (floatrows);
  }

  public Collection<L> getLabels() {
    return weights.keySet();
  }

  public Double getWeight(L key) {
    return weights.get(key);
  }

  public boolean isWeightsEmpty() {
    return weights.isEmpty();
  }

  public abstract static class Builder<L extends Comparable> {
// ------------------------------ FIELDS ------------------------------

    // these are for training only
    public double cache_size;// in MB
    public double eps;// stopping criteria
    public int maxIterations = 50000; // cap the iterations to shrink runtime
    public double nu;// for NU_SVC, ONE_CLASS, and NU_SVR
    public double p;// for EPSILON_SVR
    public boolean shrinking;// use the shrinking heuristics


    public double oneVsAllThreshold = 0.5;
    public MultiClassModel.OneVsAllMode oneVsAllMode = MultiClassModel.OneVsAllMode.None;
    public MultiClassModel.AllVsAllMode allVsAllMode = MultiClassModel.AllVsAllMode.AllVsAll;
    public double minVoteProportion;
    public int falseClassSVlimit = Integer.MAX_VALUE;

    /**
     * For unbalanced data, redistribute the misclassification cost C according to the numbers of examples in each class,
     * so that each class has the same total misclassification weight assigned to it and the average is param.C
     */
    public boolean redistributeUnbalancedC = true;


    public boolean scaleBinaryMachinesIndependently = false;
    public boolean gridsearchBinaryMachinesIndependently = false;
    public boolean normalizeL2 = false;

    /**
     * When learning scaling, only bother with this many examples, assuming they're in random order.
     */
    //	public int scalingExamples = Integer.MAX_VALUE;

    public int crossValidationFolds = 5;

    // these params are most likely to change in a copy

    public boolean probability;// do probability estimates
    // We need to maintain the labels (the key on this map) in insertion order
    public LinkedHashMap<L, Double> weights = new LinkedHashMap<L, Double>();
    public ScalingModelLearner scalingModelLearner;
    //	public boolean crossValidation;

// --------------------------- CONSTRUCTORS ---------------------------

    public Builder() {
    }

    protected Builder(ImmutableSvmParameter<L> copyFrom) {
      cache_size = copyFrom.cache_size;
      eps = copyFrom.eps;
      weights = new LinkedHashMap<>(copyFrom.weights);
      nu = copyFrom.nu;
      p = copyFrom.p;
      maxIterations = copyFrom.maxIterations;
      shrinking = copyFrom.shrinking;
      probability = copyFrom.probability;
      oneVsAllThreshold = copyFrom.oneVsAllThreshold;
      oneVsAllMode = copyFrom.oneVsAllMode;
      allVsAllMode = copyFrom.allVsAllMode;
      minVoteProportion = copyFrom.minVoteProportion;
      falseClassSVlimit = copyFrom.falseClassSVlimit;
      scaleBinaryMachinesIndependently = copyFrom.scaleBinaryMachinesIndependently;
      normalizeL2 = copyFrom.normalizeL2;
      redistributeUnbalancedC = copyFrom.redistributeUnbalancedC;
      gridsearchBinaryMachinesIndependently = copyFrom.gridsearchBinaryMachinesIndependently;
      crossValidationFolds = copyFrom.crossValidationFolds;
      scalingModelLearner = copyFrom.scalingModelLearner;
    }

    protected Builder(Builder<L> copyFrom) {
      cache_size = copyFrom.cache_size;
      eps = copyFrom.eps;
      weights = new LinkedHashMap<>(copyFrom.weights);
      nu = copyFrom.nu;
      p = copyFrom.p;
      maxIterations = copyFrom.maxIterations;
      shrinking = copyFrom.shrinking;
      probability = copyFrom.probability;
      oneVsAllThreshold = copyFrom.oneVsAllThreshold;
      oneVsAllMode = copyFrom.oneVsAllMode;
      allVsAllMode = copyFrom.allVsAllMode;
      minVoteProportion = copyFrom.minVoteProportion;
      falseClassSVlimit = copyFrom.falseClassSVlimit;
      scaleBinaryMachinesIndependently = copyFrom.scaleBinaryMachinesIndependently;
      normalizeL2 = copyFrom.normalizeL2;
      redistributeUnbalancedC = copyFrom.redistributeUnbalancedC;
      gridsearchBinaryMachinesIndependently = copyFrom.gridsearchBinaryMachinesIndependently;
      crossValidationFolds = copyFrom.crossValidationFolds;
      scalingModelLearner = copyFrom.scalingModelLearner;
    }

// -------------------------- OTHER METHODS --------------------------

    public void putWeight(L key, Double weight) {
      weights.put(key, weight);
    }

    public abstract ImmutableSvmParameter<L> build();
  }


  public ImmutableSvmParameter<L> noProbabilityCopy() {
    if (!probability) {
      return this;
    } else {
      ImmutableSvmParameter.Builder<L> builder = asBuilder();
      builder.probability = false;
      return builder.build();
    }
  }

  public ImmutableSvmParameter<L> withProbabilityCopy() {
    if (probability) {
      return this;
    } else {
      ImmutableSvmParameter.Builder<L> builder = asBuilder();
      builder.probability = true;
      return builder.build();
    }
  }

  public abstract Builder<L> asBuilder();
}
