package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ZscoreScalingModelLearner implements ScalingModelLearner {
// ------------------------------ FIELDS ------------------------------

  private final int maxExamples;
  private final boolean normalizeL2;

// -------------------------- STATIC METHODS --------------------------

  // running mean is obvious; running stddev from http://en.wikipedia.org/wiki/Standard_deviation

  public static double runningMean(int sampleCount, double priorMean, double value) {
    double d = sampleCount;  // cast only once
    return priorMean + (value - priorMean) / d;
  }

  public static double runningStddevQ(int sampleCount, double priorMean, double priorQ, double value) {
    double d = value - priorMean;
    double result = priorQ + ((sampleCount - 1) * d * d / sampleCount);
    //	assert result < 1000;  // temporary test
    //	assert !Double.isInfinite(result);
    //	assert !Double.isNaN(result);
    return result;
  }

  public static void runningStddevQtoStddevInPlace(double[] stddevQ, int sampleCount) {

    double d = sampleCount;  // cast only once
    for (int index = 0; index < stddevQ.length; index++) {
      stddevQ[index] = (double) Math.sqrt(stddevQ[index] / d);
    }
  }

// --------------------------- CONSTRUCTORS ---------------------------

  public ZscoreScalingModelLearner(int scalingExamples, boolean normalizeL2) {
    this.maxExamples = scalingExamples;
    this.normalizeL2 = normalizeL2;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModelLearner ---------------------

  public ScalingModel learnScaling(Iterable<SparseVector> examples) {
    double[] mean = null;
    double[] stddevQ = null;

    int sampleCount = 0;
    for (SparseVector example : examples) {

      if (mean == null) {
        mean = new double[example.getMaxDimensions()];
        stddevQ = new double[example.getMaxDimensions()];
      }

      if (sampleCount >= maxExamples) {
        break;
      }

      sampleCount++;  // runningMean etc. assume 1-based indexes

      for (int index : example.getIndexes()) {
        double v = example.get(index);

        double currentMean = mean[index];

        mean[index] = runningMean(sampleCount, currentMean, v);
        stddevQ[index] = runningStddevQ(sampleCount, currentMean, stddevQ[index], v);
      }

      // if an index is not seen, it's still counted as having a value of zero
    }

    runningStddevQtoStddevInPlace(stddevQ, sampleCount);
    return new ZscoreScalingModel(mean, stddevQ);
  }

// -------------------------- INNER CLASSES --------------------------

  public class ZscoreScalingModel implements ScalingModel {
// ------------------------------ FIELDS ------------------------------

    private final double[] mean;
    private final double[] stddev;

// --------------------------- CONSTRUCTORS ---------------------------

    public ZscoreScalingModel(double[] mean, double[] stddev) {
      this.mean = mean;
      this.stddev = stddev;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModel ---------------------

    public SparseVector scaledCopy(SparseVector example) {
      SparseVector result = new SparseVector(example);

      for (int i = 0; i < example.getIndexes().length; i++) {
        int index = example.getIndexes()[i];
        double v = example.getValues()[i];

        result.getIndexes()[i] = index;
        double theMean = mean[index];

        // if this dimension was never seen in the training set, then we can't scale it
        if (theMean > 0.0) {
          result.getValues()[i] = (v - theMean) / stddev[index];
        }
      }
      if (normalizeL2) {
        result.normalizeL2();
      }

      return result;
    }
  }
}
