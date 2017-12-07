package edu.berkeley.compbio.jlibsvm.util;


import org.junit.Assert;
import org.junit.Test;

public class SparseVectorTest {

  @Test
  public void testToString() {
    final SparseVector vector = new SparseVector(
        25,
        new int[]{4, 10, 19},
        new double[]{4.5f, 3.4f, 9.0f}
    );

    String value = vector.toString();

    Assert.assertEquals(value, "25+4:4.5 10:3.4 19:9.0 ");
  }

  @Test
  public void testFromString() {
    SparseVector vector = SparseVector.fromString("25+4:4.5 10:3.4 19:9.0 ");

    Assert.assertNotNull(vector);

    Assert.assertEquals(vector.getMaxDimensions(), 25);
    Assert.assertEquals(vector.getIndexes()[0], 4);
    Assert.assertEquals(vector.getIndexes()[1], 10);
    Assert.assertEquals(vector.getIndexes()[2], 19);

    Assert.assertEquals(vector.getValues()[0], 4.5f, 0.01);
    Assert.assertEquals(vector.getValues()[1], 3.4f, 0.01);
    Assert.assertEquals(vector.getValues()[2], 9.0f, 0.01);
  }

  @Test
  public void testToDenseVector() {
    final SparseVector vector1 = new SparseVector(
        5,
        new int[]{2, 3, 4},
        new double[]{4.5f, 3.4f, 9.0f}
    );

    double[] dv = vector1.toDenseVector();

    Assert.assertNotNull(dv);
    Assert.assertEquals(5, dv.length);

    Assert.assertEquals(0.0f, dv[0], 0.01);
    Assert.assertEquals(0.0f, dv[1], 0.01);
    Assert.assertEquals(4.5f, dv[2], 0.01);
    Assert.assertEquals(3.4f, dv[3], 0.01);
    Assert.assertEquals(9.0f, dv[4], 0.01);
  }

  @Test
  public void testDot() {
    final SparseVector vector1 = new SparseVector(
        5,
        new int[]{2, 3, 4},
        new double[]{4.5f, 3.4f, 9.0f}
    );

    final SparseVector vector2 = new SparseVector(
        5,
        new int[]{1, 3, 4},
        new double[]{1.5f, 7.6f, 5.7f}
    );

    double value = SparseVector.dot(vector1, vector2);

    Assert.assertEquals(value,
        1.5f * 0.0f + 4.5f * 0.0f + 3.4f * 7.6f + 9.0f * 5.7f, 0.01);
  }

  @Test
  public void testDifference() {
    final SparseVector vector1 = new SparseVector(
        5,
        new int[]{2, 3, 4},
        new double[]{4.5f, 3.4f, 9.0f}
    );

    final SparseVector vector2 = new SparseVector(
        5,
        new int[]{1, 3, 4},
        new double[]{1.5f, 7.6f, 5.7f}
    );

    SparseVector diff = SparseVector.difference(vector1, vector2);

    Assert.assertNotNull(diff);

    Assert.assertEquals(diff.getValues().length, 4);
    Assert.assertEquals(diff.getValues()[0], 0.0f - 1.5f, 0.01);
    Assert.assertEquals(diff.getValues()[1], 4.5f - 0.0f, 0.01);
    Assert.assertEquals(diff.getValues()[2], 3.4f - 7.6f, 0.01);
    Assert.assertEquals(diff.getValues()[3], 9.0f - 5.7f, 0.01);
  }

  @Test
  public void testSquareNorm() {
    final SparseVector vector1 = new SparseVector(
        5,
        new int[]{2, 3, 4},
        new double[]{4.5f, 3.4f, 9.0f}
    );

    final SparseVector vector2 = new SparseVector(
        5,
        new int[]{1, 3, 4},
        new double[]{1.5f, 7.6f, 5.7f}
    );

    final double val = SparseVector.squareNorm(vector1, vector2);

    Assert.assertEquals(51.02999965667732f, val, 0.01);
  }

  @Test
  public void testOf() {
    double[] d1 = new double[]{0.0, 0.1, -0.3, 0.0, 10.0};

    final SparseVector dv = SparseVector.of(d1);

    Assert.assertNotNull(dv);

    Assert.assertEquals(5, dv.getMaxDimensions());
    Assert.assertEquals(3, dv.getIndexes().length);

    Assert.assertEquals(0.1, dv.getValues()[0], 0.01);
    Assert.assertEquals(-0.3, dv.getValues()[1], 0.01);
    Assert.assertEquals(10.0, dv.getValues()[2], 0.01);
  }
}