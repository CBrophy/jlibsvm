package edu.berkeley.compbio.jlibsvm.scaler;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * Learn the minima and maxima of each dimension from the training data, so as to transform points into the [-1, 1]
 * interval.  Test points that lie outside the bounds given by the training data will have values lying outside this
 * interval.
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class LinearScalingModelLearner implements ScalingModelLearner<SparseVector> {
// ------------------------------ FIELDS ------------------------------

  //ImmutableSvmParameter param;
  private final int maxExamples;
  private final boolean normalizeL2;

// --------------------------- CONSTRUCTORS ---------------------------

  public LinearScalingModelLearner(int scalingExamples, boolean normalizeL2) {
    this.maxExamples = scalingExamples;
    this.normalizeL2 = normalizeL2;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModelLearner ---------------------

  public ScalingModel<SparseVector> learnScaling(Iterable<SparseVector> examples) {
    double[] minima = null;
    double[] sizes = null;

    int count = 0;
    for (SparseVector example : examples) {

      if (minima == null) {
        minima = new double[example.getMaxDimensions()];
        sizes = new double[example.getMaxDimensions()];
      }

      if (count >= maxExamples) {
        break;
      }

      for (int index : example.getIndexes()) {
        double v = example.get(index);

        minima[index] = Math.min(minima[index], v);
        sizes[index] = Math.max(sizes[index], v - minima[index]);

      }
      count++;
    }

    return new LinearScalingModel(minima, sizes);
  }

// -------------------------- INNER CLASSES --------------------------

  public class LinearScalingModel implements ScalingModel<SparseVector> {
// ------------------------------ FIELDS ------------------------------

    double[] minima;
    //double[] maxima;
    double[] sizes;

// --------------------------- CONSTRUCTORS ---------------------------

    public LinearScalingModel(double[] minima, double[] sizes) {
      this.minima = minima;
      this.sizes = sizes;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ScalingModel ---------------------

    public SparseVector scaledCopy(SparseVector example) {
      SparseVector result = new SparseVector(example);

      for (int i = 0; i < example.getIndexes().length; i++) {
        int index = example.getIndexes()[i];
        double v = example.getValues()[i];

        result.getIndexes()[i] = index;
        double min = minima[index];

        // if this dimension was never seen in the training set, then we can't scale it
        if (sizes[index] > 0.0f) {
          result.getValues()[i] = (2F * (v - min) / sizes[index]) - 1F;
        }

      }

      if (normalizeL2) {
        result.normalizeL2();
      }

      return result;
    }
  }
}
