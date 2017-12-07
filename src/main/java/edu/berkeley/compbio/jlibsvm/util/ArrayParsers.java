package edu.berkeley.compbio.jlibsvm.util;

import java.util.StringTokenizer;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class ArrayParsers {
// -------------------------- STATIC METHODS --------------------------

  public static double[] parseFloatArray(String s) {
    StringTokenizer st = new StringTokenizer(s);
    double[] result = new double[st.countTokens()];
    int i = 0;
    while (st.hasMoreTokens()) {
      result[i] = Double.parseDouble(st.nextToken());
      i++;
    }
    return result;
  }

  public static int[] parseIntArray(String s) {
    StringTokenizer st = new StringTokenizer(s);
    int[] result = new int[st.countTokens()];
    int i = 0;
    while (st.hasMoreTokens()) {
      result[i] = Integer.parseInt(st.nextToken());
      i++;
    }
    return result;
  }
}
