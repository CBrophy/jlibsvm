package edu.berkeley.compbio.jlibsvm.oneclass;

import edu.berkeley.compbio.jlibsvm.DiscreteModel;
import edu.berkeley.compbio.jlibsvm.regression.RegressionModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class OneClassModel<L, P extends SparseVector> extends RegressionModel<P> implements DiscreteModel<Boolean, P> {
// ------------------------------ FIELDS ------------------------------

  L label;

// --------------------------- CONSTRUCTORS ---------------------------

  public OneClassModel() {
    super();
  }


// --------------------- GETTER / SETTER METHODS ---------------------

  public L getLabel() {
    return label;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ContinuousModel ---------------------

  //** Hmmm does this make sense?

  public Double predictValue(P x) {
    return predictLabel(x) ? 1.0 : -1.0;
  }

// --------------------- Interface DiscreteModel ---------------------

  public Boolean predictLabel(P x) {
    return super.predictValue(x) > 0;
  }

// -------------------------- OTHER METHODS --------------------------

  /**
   * HACK guess at a probability of being in the one class by logistic function.  To be fancier we could do some sigmoid
   * thing, and take the laplace parameter into account, etc.  Is it valid to do cross-validation and train a sigmoid
   * model just like in C-SVC?
   */
  public double getProbability(P x) {
    // REVIEW one-class probability hack
    // at least the logistic function is monotonic
    double v = super.predictValue(x);
    double result = 1 / (1 + Math.exp(-v));
    return (double) result;

  }
}
