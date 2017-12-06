package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ZscoreScalingModelLearner implements ScalingModelLearner<SparseVector> {
// ------------------------------ FIELDS ------------------------------

  private final int maxExamples;
  private final boolean normalizeL2;

// -------------------------- STATIC METHODS --------------------------

  // running mean is obvious; running stddev from http://en.wikipedia.org/wiki/Standard_deviation

  public static float runningMean(int sampleCount, float priorMean, float value) {
    float d = sampleCount;  // cast only once
    return priorMean + (value - priorMean) / d;
  }

  public static float runningStddevQ(int sampleCount, float priorMean, float priorQ, float value) {
    float d = value - priorMean;
    float result = priorQ + ((sampleCount - 1) * d * d / sampleCount);
    //	assert result < 1000;  // temporary test
    //	assert !Float.isInfinite(result);
    //	assert !Float.isNaN(result);
    return result;
  }

  public static void runningStddevQtoStddevInPlace(float[] stddevQ, int sampleCount) {

    float d = sampleCount;  // cast only once
    for (int index = 0; index < stddevQ.length; index++) {
      stddevQ[index] = (float) Math.sqrt(stddevQ[index] / d);
    }
  }

// --------------------------- CONSTRUCTORS ---------------------------

  public ZscoreScalingModelLearner(int scalingExamples, boolean normalizeL2) {
    this.maxExamples = scalingExamples;
    this.normalizeL2 = normalizeL2;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModelLearner ---------------------

  public ScalingModel<SparseVector> learnScaling(Iterable<SparseVector> examples) {
    float[] mean = null;
    float[] stddevQ = null;

    int sampleCount = 0;
    for (SparseVector example : examples) {

      if (mean == null) {
        mean = new float[example.getMaxDimensions()];
        stddevQ = new float[example.getMaxDimensions()];
      }

      if (sampleCount >= maxExamples) {
        break;
      }

      sampleCount++;  // runningMean etc. assume 1-based indexes

      for (int index : example.getIndexes()) {
        float v = example.get(index);

        float currentMean = mean[index];

        mean[index] = runningMean(sampleCount, currentMean, v);
        stddevQ[index] = runningStddevQ(sampleCount, currentMean, stddevQ[index], v);
      }

      // if an index is not seen, it's still counted as having a value of zero
    }

    runningStddevQtoStddevInPlace(stddevQ, sampleCount);
    return new ZscoreScalingModel(mean, stddevQ);
  }

// -------------------------- INNER CLASSES --------------------------

  public class ZscoreScalingModel implements ScalingModel<SparseVector> {
// ------------------------------ FIELDS ------------------------------

    private final float[] mean;
    private final float[] stddev;

// --------------------------- CONSTRUCTORS ---------------------------

    public ZscoreScalingModel(float[] mean, float[] stddev) {
      this.mean = mean;
      this.stddev = stddev;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModel ---------------------

    public SparseVector scaledCopy(SparseVector example) {
      SparseVector result = new SparseVector(example);

      for (int i = 0; i < example.getIndexes().length; i++) {
        int index = example.getIndexes()[i];
        float v = example.getValues()[i];

        result.getIndexes()[i] = index;
        float theMean = mean[index];

        // if this dimension was never seen in the training set, then we can't scale it
        if (theMean > 0.0f) {
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
