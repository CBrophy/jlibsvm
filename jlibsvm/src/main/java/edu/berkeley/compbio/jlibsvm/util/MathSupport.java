package edu.berkeley.compbio.jlibsvm.util;

import java.io.Serializable;

public class MathSupport implements Serializable {
// -------------------------- STATIC METHODS --------------------------

  public static double powi(double base, int times) {
    assert times >= 0;
    double tmp = base, ret = 1.0f;

    for (int t = times; t > 0; t /= 2) {
      if (t % 2 != 0) {
        ret *= tmp;
      }
      tmp = tmp * tmp;
    }
    return ret;
  }

  /**
   * This is provided by apache commons, but let's avoid the dependency
   */
  public static boolean[] toPrimitive(Boolean[] x) {
    boolean[] result = new boolean[x.length];
    int i = 0;
    for (Boolean b : x) {
      result[i] = b;
      i++;
    }
    return result;
  }
}
