package edu.berkeley.compbio.jlibsvm.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

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

  public SparseVector(int maxDimensions) {
    this(
        maxDimensions,
        new int[maxDimensions],
        new float[maxDimensions]
    );
  }

  public SparseVector(SparseVector sparseVector) {
    this(
        sparseVector.getMaxDimensions(),
        Arrays.copyOf(sparseVector.indexes, sparseVector.indexes.length),
        Arrays.copyOf(sparseVector.values, sparseVector.values.length)
    );
  }

  public int getMaxDimensions() {
    return maxDimensions;
  }

  public int[] getIndexes() {
    return indexes;
  }

  public float[] getValues() {
    return values;
  }

  public SparseVector(final int maxDimensions, final int nonZeroDimensions) {
    this(
        maxDimensions,
        new int[nonZeroDimensions],
        new float[nonZeroDimensions]
    );
  }

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
      int valueIndex = currentIndex < this.indexes.length ? this.indexes[currentIndex] : -1;

      result[index] = 0.0f;

      if (valueIndex == index) {
        result[index] = values[currentIndex];
        currentIndex++;
      }

    }

    return result;
  }

  public int maxIndex() {
    return indexes[indexes.length - 1];
  }

  public static SparseVector mergeScaleVectors(
      SparseVector sv1, float p1, SparseVector sv2, float p2
  ) {
    // need the resulting indexes to be sorted; just brute force through the possible indexes
    // note this works for sparse subclasses that e.g. provide a default value

    int maxDimensions = Math.max(sv1.getMaxDimensions(), sv2.getMaxDimensions());

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

  @Override
  public String toString() {
    StringBuffer sb = new StringBuffer();
    sb.append(maxDimensions).append('+');
    for (int j = 0; j < indexes.length; j++) {
      sb.append(indexes[j] + ":" + values[j] + ' ');
    }
    return sb.toString();
  }

  public static SparseVector fromString(final String vectorString) {
    assert vectorString != null;
    final String trimmed = vectorString.trim();

    if (trimmed.isEmpty()) {
      return null;
    }

    int plus = trimmed.indexOf('+');
    if (plus < 0) {
      throw new RuntimeException("Malformed vector string: " + vectorString);
    }

    int maxDimensions = Integer.parseInt(trimmed.substring(0, plus));

    StringTokenizer tokenizer = new StringTokenizer(trimmed.substring(plus + 1));

    final List<Integer> indexList = new ArrayList<>(maxDimensions);
    final List<Float> valueList = new ArrayList<>(maxDimensions);

    while (tokenizer.hasMoreTokens()) {
      String vectorParts = tokenizer.nextToken();
      int colonIndex = vectorParts.indexOf(':');

      if (colonIndex < 0) {
        continue;
      }

      Integer index = Integer.parseInt(vectorParts.substring(0, colonIndex));
      Float value = Float.parseFloat(vectorParts.substring(colonIndex + 1));

      indexList.add(index);
      valueList.add(value);
    }

    int[] indices = new int[indexList.size()];
    float[] values = new float[indexList.size()];

    for (int index = 0; index < indexList.size(); index++) {
      indices[index] = indexList.get(index);
      values[index] = valueList.get(index);
    }

    return new SparseVector(
        maxDimensions,
        indices,
        values
    );
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

  public static double dot(final SparseVector x, final SparseVector y) {
    assert x != null;
    assert y != null;
    assert x.maxDimensions == y.maxDimensions;

    if(x.indexes.length == 0) {
      return 0.0;
    }

    double dotProduct = 0.0;
    int currentXIndex = 0;
    int currentYIndex = 0;

    for (int index = 0; index < x.getMaxDimensions(); index++) {
      int xIndex = currentXIndex < x.indexes.length ? x.indexes[currentXIndex] : -1;
      int yIndex = currentYIndex < y.indexes.length ? y.indexes[currentYIndex] : -1;
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

  public static double squareNorm(final SparseVector x, final SparseVector y){
    SparseVector diff = SparseVector.difference(x, y);

    return SparseVector.dot(diff, diff);
  }

  public static SparseVector difference(final SparseVector v1, final SparseVector v2) {
    assert v1 != null;
    assert v2 != null;
    assert v1.getMaxDimensions() == v2.getMaxDimensions();

    List<Integer> indexList = new ArrayList<>(v1.maxDimensions);
    List<Float> valueList = new ArrayList<>(v1.maxDimensions);

    int currentIndex1 = 0;
    int currentIndex2 = 0;

    for(int index = 0; index < v1.getMaxDimensions(); index++){

      int valIndex1 = currentIndex1 < v1.indexes.length ? v1.indexes[currentIndex1] : -1;
      int valIndex2 = currentIndex2 < v2.indexes.length ? v2.indexes[currentIndex2] : -1;

      float val1 = 0.0f;
      float val2 = 0.0f;

      if(valIndex1 == index){
        val1 = v1.values[currentIndex1];
        currentIndex1++;
      }

      if(valIndex2 == index){
        val2 = v2.values[currentIndex2];
        currentIndex2++;
      }

      float val = val1 - val2;

      if(val > 0.0f || val < 0.0f){
        indexList.add(index);
        valueList.add(val);
      }
    }

    int[] indices = new int[indexList.size()];
    float[] values = new float[indexList.size()];

    for(int index = 0; index < indexList.size(); index++){
      indices[index] = indexList.get(index);
      values[index] = valueList.get(index);
    }

    return new SparseVector(v1.getMaxDimensions(), indices, values);
  }

  // -------------------------- float iterator --------------------------
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
      int nextIndex = currentIndex < sparseVector.indexes.length ? sparseVector.indexes[currentIndex] : -1;

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

}
