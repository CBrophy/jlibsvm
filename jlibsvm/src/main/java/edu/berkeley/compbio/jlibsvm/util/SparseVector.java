package edu.berkeley.compbio.jlibsvm.util;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.StringTokenizer;

/**
 * This acts something like a Map from int to double
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SparseVector implements Serializable {
// ------------------------------ FIELDS ------------------------------

  private final Long id;
  private final int maxDimensions;
  private final int[] indexes;
  private final double[] values;
  private final int hashCode;

// --------------------------- CONSTRUCTORS ---------------------------

  SparseVector(
      final int maxDimensions,
      final int[] indexes,
      final double[] values) {
    this(null, maxDimensions, indexes, values);
  }

  SparseVector(
      final Long id,
      final int maxDimensions,
      final int[] indexes,
      final double[] values) {
    assert indexes != null;
    assert values != null;
    assert maxDimensions >= indexes.length;

    this.id = id;
    this.maxDimensions = maxDimensions;
    this.indexes = indexes;
    this.values = values;
    this.hashCode = calcHashCode(
        id,
        maxDimensions,
        indexes,
        values
    );
  }

  private static int calcHashCode(
      final Long id,
      final int maxDimensions,
      final int[] indexes,
      final double[] values
  ) {

    // Using the ids should allow for dupes in a map/hash scenario
    if (id != null) {
      return id.hashCode();
    }

    return Objects.hash(
        maxDimensions,
        Arrays.hashCode(indexes),
        Arrays.hashCode(values)
    );
  }

  public SparseVector(int maxDimensions) {
    this(
        maxDimensions,
        new int[maxDimensions],
        new double[maxDimensions]
    );
  }

  public SparseVector(SparseVector sparseVector) {
    this(
        sparseVector.getId(),
        sparseVector.getMaxDimensions(),
        Arrays.copyOf(sparseVector.indexes, sparseVector.indexes.length),
        Arrays.copyOf(sparseVector.values, sparseVector.values.length)
    );
  }

  public long getId() {
    return id;
  }

  public int getMaxDimensions() {
    return maxDimensions;
  }

  public int[] getIndexes() {
    return indexes;
  }

  public double[] getValues() {
    return values;
  }

  public SparseVector(final int maxDimensions, final int nonZeroDimensions) {
    this(
        maxDimensions,
        new int[nonZeroDimensions],
        new double[nonZeroDimensions]
    );
  }

  /**
   * Create randomized vectors for testing
   */
  public static SparseVector createRandomSparseVector(
      int maxDimensions,
      double nonzeroProbability,
      double maxValue) {
    List<Integer> indexList = new ArrayList<Integer>(maxDimensions);

    for (int i = 0; i < maxDimensions; i++) {
      if (Math.random() < (double) nonzeroProbability) {
        indexList.add(i);
      }
    }

    final SparseVector result = new SparseVector(maxDimensions, indexList.size());
    for (int i = 0; i < indexList.size(); i++) {
      result.indexes[i] = indexList.get(i);
      result.values[i] = (double) (Math.random() * maxValue);
    }
    return result;
  }

  public Iterator<Double> getIterator() {
    return new SparseVectorIterator(this);
  }

  public double[] toDenseVector() {
    int currentIndex = 0;
    double[] result = new double[maxDimensions];

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
      SparseVector sv1, double p1, SparseVector sv2, double p2
  ) {
    // need the resulting indexes to be sorted; just brute force through the possible indexes
    // note this works for sparse subclasses that e.g. provide a default value

    int maxDimensions = Math.max(sv1.getMaxDimensions(), sv2.getMaxDimensions());

    List<Integer> indexList = new ArrayList<>();
    List<Double> valueList = new ArrayList<>();

    for (int i = 0; i < maxDimensions; i++) {
      double v = sv1.get(i) * p1 + sv2.get(i) * p2;
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
    return of(null, denseVector);
  }

  public static SparseVector of(final Long id, final double[] denseVector) {
    List<Integer> indexList = new ArrayList<>(denseVector.length);
    List<Double> valueList = new ArrayList<>(denseVector.length);

    for (int index = 0; index < denseVector.length; index++) {
      if (denseVector[index] > 0.0d || denseVector[index] < 0.0d) {
        indexList.add(index);
        valueList.add(denseVector[index]);
      }
    }

    final int[] indexes = new int[indexList.size()];
    final double[] values = new double[valueList.size()];

    for (int index = 0; index < indexList.size(); index++) {
      indexes[index] = indexList.get(index);
      values[index] = valueList.get(index);
    }

    return new SparseVector(
        id,
        denseVector.length,
        indexes,
        values);

  }

  public double get(int i) {
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

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof SparseVector
        && getId() == ((SparseVector) obj).getId()
        && getMaxDimensions() == ((SparseVector) obj).getMaxDimensions()
        && Arrays.equals(getIndexes(), ((SparseVector) obj).getIndexes())
        && Arrays.equals(getValues(), ((SparseVector) obj).getValues());
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
    final List<Double> valueList = new ArrayList<>(maxDimensions);

    while (tokenizer.hasMoreTokens()) {
      String vectorParts = tokenizer.nextToken();
      int colonIndex = vectorParts.indexOf(':');

      if (colonIndex < 0) {
        continue;
      }

      Integer index = Integer.parseInt(vectorParts.substring(0, colonIndex));
      Double value = Double.parseDouble(vectorParts.substring(colonIndex + 1));

      indexList.add(index);
      valueList.add(value);
    }

    int[] indices = new int[indexList.size()];
    double[] values = new double[indexList.size()];

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
    for (double value : values) {
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

    if (x.indexes.length == 0) {
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

  public static double squareNorm(final SparseVector x, final SparseVector y) {
    SparseVector diff = SparseVector.difference(x, y);

    return SparseVector.dot(diff, diff);
  }

  public static SparseVector difference(final SparseVector v1, final SparseVector v2) {
    assert v1 != null;
    assert v2 != null;
    assert v1.getMaxDimensions() == v2.getMaxDimensions();

    List<Integer> indexList = new ArrayList<>(v1.maxDimensions);
    List<Double> valueList = new ArrayList<>(v1.maxDimensions);

    int currentIndex1 = 0;
    int currentIndex2 = 0;

    for (int index = 0; index < v1.getMaxDimensions(); index++) {

      int valIndex1 = currentIndex1 < v1.indexes.length ? v1.indexes[currentIndex1] : -1;
      int valIndex2 = currentIndex2 < v2.indexes.length ? v2.indexes[currentIndex2] : -1;

      double val1 = 0.0f;
      double val2 = 0.0f;

      if (valIndex1 == index) {
        val1 = v1.values[currentIndex1];
        currentIndex1++;
      }

      if (valIndex2 == index) {
        val2 = v2.values[currentIndex2];
        currentIndex2++;
      }

      double val = val1 - val2;

      if (val > 0.0f || val < 0.0f) {
        indexList.add(index);
        valueList.add(val);
      }
    }

    int[] indices = new int[indexList.size()];
    double[] values = new double[indexList.size()];

    for (int index = 0; index < indexList.size(); index++) {
      indices[index] = indexList.get(index);
      values[index] = valueList.get(index);
    }

    return new SparseVector(v1.getMaxDimensions(), indices, values);
  }

  // -------------------------- double iterator --------------------------
  private static class SparseVectorIterator implements Iterator<Double> {

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
    public Double next() {
      int nextIndex =
          currentIndex < sparseVector.indexes.length ? sparseVector.indexes[currentIndex] : -1;

      double result = 0.0f;

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
