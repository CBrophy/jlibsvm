package edu.berkeley.compbio.jlibsvm.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * This acts something like a Map from int to float
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SparseVector implements Serializable {
// ------------------------------ FIELDS ------------------------------

  private final int maxDimensions;
  private final int[] indexes;
  private final float[] values;

// --------------------------- CONSTRUCTORS ---------------------------

  SparseVector(
      final int maxDimensions,
      final int[] indexes,
      final float[] values) {
    assert indexes != null;
    assert values != null;
    assert maxDimensions > indexes.length;

    this.maxDimensions = maxDimensions;
    this.indexes = indexes;
    this.values = values;
  }

  public int getMaxDimensions() {
    return maxDimensions;
  }

  public SparseVector(final int maxDimensions, final int nonZeroDimensions) {
    this(
        maxDimensions,
        new int[nonZeroDimensions],
        new float[nonZeroDimensions]
    );
  }

  // don't want to put this here since it assumes the usual dot product, but we might want a different one
    /*
     private static final float NOT_COMPUTED_YET = -1;
     private float normSquared = NOT_COMPUTED_YET;

     public float getNormSquared()
         {
         if (normSquared == -1)
             {
             normSquared = 0;
             int xlen = values.length;
             int i = 0;
             while (i < xlen)
                 {
                 normSquared += values[i] * values[i];
                 ++i;
                 }
             }
         return normSquared;
         }*/

  /**
   * Create randomized vectors for testing
   */
  public static SparseVector createRandomSparseVector(
      int maxDimensions,
      float nonzeroProbability,
      float maxValue) {
    List<Integer> indexList = new ArrayList<Integer>(maxDimensions);

    for (int i = 0; i < maxDimensions; i++) {
      if (Math.random() < (double) nonzeroProbability) {
        indexList.add(i);
      }
    }

    final SparseVector result = new SparseVector(maxDimensions, indexList.size());
    for (int i = 0; i < indexList.size(); i++) {
      result.indexes[i] = indexList.get(i);
      result.values[i] = (float) (Math.random() * maxValue);
    }
    return result;
  }

  public Iterator<Float> getIterator() {
    return new SparseVectorIterator(this);
  }

  public float[] toDenseVector() {
    int currentIndex = 0;
    float[] result = new float[maxDimensions];

    for (int index = 0; index < this.maxDimensions; index++) {
      int valueIndex = this.indexes[currentIndex];

      result[index] = 0.0f;

      if (valueIndex == index) {
        result[index] = values[valueIndex];
      }

    }

    return result;
  }

  public int maxIndex() {
    return indexes[indexes.length - 1];
  }

  public static SparseVector createSparseVector(
      SparseVector sv1, float p1, SparseVector sv2, float p2
  ) {
    // need the resulting indexes to be sorted; just brute force through the possible indexes
    // note this works for sparse subclasses that e.g. provide a default value

    int maxDimensions = Math.max(sv1.maxIndex(), sv2.maxIndex());

    List<Integer> indexList = new ArrayList<Integer>();
    List<Float> valueList = new ArrayList<Float>();

    for (int i = 0; i < maxDimensions; i++) {
      float v = sv1.get(i) * p1 + sv2.get(i) * p2;
      if (v != 0) {
        indexList.add(i);
        valueList.add(v);
      }
    }

    final SparseVector result = new SparseVector(
        maxDimensions,
        indexList.size()
    );

    for (int i = 0; i < indexList.size(); i++) {
      result.indexes[i] = indexList.get(i);
      result.values[i] = valueList.get(i);
    }

    return result;
  }

  public static SparseVector of(final double[] denseVector) {
    List<Integer> indexList = new ArrayList<>(denseVector.length);
    List<Float> valueList = new ArrayList<>(denseVector.length);

    for (int index = 0; index < denseVector.length; index++) {
      if (denseVector[index] > 0.0) {
        indexList.add(index);
        valueList.add((float) denseVector[index]);
      }
    }

    final int[] indexes = new int[indexList.size()];
    final float[] values = new float[valueList.size()];

    return new SparseVector(
        denseVector.length,
        indexes,
        values);

  }

  public static SparseVector of(final float[] denseVector) {
    List<Integer> indexList = new ArrayList<>(denseVector.length);
    List<Float> valueList = new ArrayList<>(denseVector.length);

    for (int index = 0; index < denseVector.length; index++) {
      if (denseVector[index] > 0.0) {
        indexList.add(index);
        valueList.add(denseVector[index]);
      }
    }

    final int[] indexes = new int[indexList.size()];
    final float[] values = new float[valueList.size()];

    return new SparseVector(
        denseVector.length,
        indexes,
        values);

  }

  public float get(int i) {
    int j = Arrays.binarySearch(indexes, i);
    if (j < 0) {
      return 0;
    } else {
      return values[j];
    }
  }

// ------------------------ CANONICAL METHODS ------------------------

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for (int j = 0; j < indexes.length; j++) {
      sb.append(indexes[j] + ":" + values[j] + " ");
    }
    return sb.toString();
  }

// -------------------------- OTHER METHODS --------------------------

  public void normalizeL2() {
    double sumOfSquares = 0;
    for (float value : values) {
      sumOfSquares += value * value;
    }

    double total = Math.sqrt(sumOfSquares);
    for (int i = 0; i < values.length; i++) {
      values[i] /= total;
    }
  }

  private static class SparseVectorIterator implements Iterator<Float> {

    private final SparseVector sparseVector;
    private int iterationIndex = 0;
    private int currentIndex = 0;

    public SparseVectorIterator(SparseVector sparseVector) {
      this.sparseVector = sparseVector;
    }

    @Override
    public boolean hasNext() {
      return iterationIndex < sparseVector.maxDimensions;
    }

    @Override
    public Float next() {
      int nextIndex = sparseVector.indexes[currentIndex];
      float result = 0.0f;
      if (nextIndex == iterationIndex) {
        result = sparseVector.values[currentIndex];
        currentIndex++;
      }
      iterationIndex++;
      return result;
    }

    @Override
    public void remove() {
      throw new RuntimeException("Not implemented");
    }
  }

  public static double dot(final SparseVector x, final SparseVector y) {
    assert x != null;
    assert y != null;
    assert x.maxDimensions == y.maxDimensions;

    double dotProduct = 0.0;
    int currentXIndex = 0;
    int currentYIndex = 0;

    for (int index = 0; index < x.getMaxDimensions(); index++) {
      int xIndex = x.indexes[currentXIndex];
      int yIndex = y.indexes[currentYIndex];
      double xValue = 0.0;
      double yValue = 0.0;

      if (xIndex == index) {
        xValue = x.values[currentXIndex];
        currentXIndex++;
      }

      if (yIndex == index) {
        yValue = y.values[currentYIndex];
        currentYIndex++;
      }

      dotProduct += xValue * yValue;
    }

    return dotProduct;
  }
}
