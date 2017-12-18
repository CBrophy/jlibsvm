package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.SolutionModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class AlphaModel<L extends Comparable> extends SolutionModel<L> implements
    Serializable {
// ------------------------------ FIELDS ------------------------------

  // used only during training, then ditched
  public Map<SparseVector, Double> supportVectors;

  // more compact representation used after training
  public int numSVs;
  public SparseVector[] SVs;
  public double[] alphas;

  public double rho;

// --------------------------- CONSTRUCTORS ---------------------------

  protected AlphaModel() {
    super();
  }

// -------------------------- OTHER METHODS --------------------------

  /**
   * Remove vectors whose alpha is zero, leaving only support vectors
   */
  public void compact() {
    // do this first so as to make the arrays the right size below
    supportVectors.entrySet().removeIf(entry -> entry.getValue() == 0);

    // put the keys and values in parallel arrays, to free memory and maybe make things a bit faster (?)

    numSVs = supportVectors.size();
    SVs = new SparseVector[numSVs];
    alphas = new double[numSVs];

    int c = 0;
    for (Map.Entry<SparseVector, Double> entry : supportVectors.entrySet()) {
      SVs[c] = entry.getKey();
      alphas[c] = entry.getValue();
      c++;
    }

    supportVectors = null;
  }

}
