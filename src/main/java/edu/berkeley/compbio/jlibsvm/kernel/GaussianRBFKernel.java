package edu.berkeley.compbio.jlibsvm.kernel;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.Properties;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class GaussianRBFKernel extends GammaKernel<SparseVector> {
// -------------------------- STATIC METHODS --------------------------

	/*	public String perfString()
		 {
		 return "" + evaluateCount + " evaluations, " + interpolatingExp.perfString();
		 }
 */
/*	private float float2xDotProduct(SvmPoint x, SvmPoint y)
		{
		// this ends up horribly wrong near the boundaries... ???
		// or not, and I was previously worried about the exp method?
		float sum = -2f * MathSupport.dot(x, y);

		sum += x.getSquared();
		sum += y.getSquared();

		return sum;
		}
*/

/*		private float float2xDotProduct(SvmPoint x, SvmPoint y)
		 {
		 // this ends up horribly wrong near the boundaries... ???
		 // or not, and I was previously worried about the exp method?
		 float sum = -2f * MathSupport.dot(x, y);

		 sum += x.getSquared();
		 sum += y.getSquared();

		 return sum;
		 }
*/

  /**
   * Subtract one vector from the other and take the dot product of the difference with itself, to get the square of the
   * norm.
   *
   * @param x
   * @param y
   * @return
   */
/*	private float explicitFloatSum(SvmPoint x, SvmPoint y)
		{
		float sum = 0;
		int xlen = x.indexes.length;
		int ylen = y.indexes.length;
		int i = 0;
		int j = 0;
		while (i < xlen && j < ylen)
			{
			if (x.indexes[i] == y.indexes[j])
				{
				float d = x.values[i++] - y.values[j++];
				sum += d * d;
				}
			else if (x.indexes[i] > y.indexes[j])
				{
				// there is an entry for y but not for x at this index => x.value == 0
				sum += y.values[j] * y.values[j];
				++j;
				}
			else
				{
				// there is an entry for x but not for y at this index => y.value == 0
				sum += x.values[i] * x.values[i];
				++i;
				}
			}

		// finish off any trailing entries in one vector but not the other
		while (i < xlen)
			{
			sum += x.values[i] * x.values[i];
			++i;
			}

		while (j < ylen)
			{
			sum += y.values[j] * y.values[j];
			++j;
			}
		return sum;
		}*/

// --------------------------- CONSTRUCTORS ---------------------------

  public GaussianRBFKernel(Properties props) {
    this(Float.parseFloat(props.getProperty("gamma")));
  }

  public GaussianRBFKernel(float gamma) {
    super(gamma);
  }

  // ------------------------ CANONICAL METHODS ------------------------

  @Override
  public String toString() {
    return "RBF gamma=" + gamma;
  }

  // BAD file output infrastructure

  public String toFileOutputString() {
    StringBuilder sb = new StringBuilder();
    sb.append("kernel_type rbf\n");
    sb.append("gamma " + gamma + "\n");
    return sb.toString();
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface KernelFunction ---------------------

  public double evaluate(@NotNull SparseVector x, @NotNull SparseVector y) {
    // try doing the internal stuff at double precision

    return Math.exp(-gamma * SparseVector.squareNorm(x, y));

  }
}
