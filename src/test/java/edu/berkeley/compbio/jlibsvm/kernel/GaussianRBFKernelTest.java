package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import org.apache.log4j.Logger;
import org.junit.Assert;
import org.junit.Test;


/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class GaussianRBFKernelTest {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(GaussianRBFKernelTest.class);

// -------------------------- OTHER METHODS --------------------------

  @Test
  public void explicitAndCompositeKernelsAreEqual() {
    float gamma = 1f;

    CompositeGaussianRBFKernel<SparseVector> composite =
        new CompositeGaussianRBFKernel<SparseVector>(gamma, new LinearKernel());

    GaussianRBFKernel explicit = new GaussianRBFKernel(gamma);

    // need a lot of iterations to use enough time for profiling (e.g. 1000)

    for (int i = 0; i < 100; i++) {
      SparseVector sv1 = SparseVector.createRandomSparseVector(100, .5f, 1);
      SparseVector sv2 = SparseVector.createRandomSparseVector(100, .5f, 1);

      // those vectors are likely far apart, and the RBF is always near zero for those.  Interpolate to test closer distances.

      for (int j = 0; j < 100; j++) {
        SparseVector sv3 = SparseVector.mergeScaleVectors(sv1, 1f - (j / 1000f), sv2, (j / 1000f));
        final double compositeResult = composite.evaluate(sv1, sv3);
        final double explicitResult = explicit.evaluate(sv1, sv3);
        final double diff = Math.abs(explicitResult - compositeResult);

        Assert.assertTrue(String.format("%s < %s", diff, 1e-8),diff < 1e-8);
      }
    }
  }

}
