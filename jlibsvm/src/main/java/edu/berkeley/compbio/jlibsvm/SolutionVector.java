package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class SolutionVector implements Comparable<SolutionVector>, Serializable{
// ------------------------------ FIELDS ------------------------------

  /**
   * Used by the cacheing mechanism to keep track of which SVs are the most active.
   */
  public int rank = -1;

  /**
   * keep track of the sample id for mapping to ranks
   */
  final public long id;
  final public SparseVector point;
  public boolean targetValue;
  public double alpha;
  public double G;
  public double linearTerm;
  Status alphaStatus;
  double G_bar;

// --------------------------- CONSTRUCTORS ---------------------------

  public SolutionVector(long id, @NotNull SparseVector key, Boolean targetValue, double linearTerm) {
    this.id = id;
    point = key;
    this.linearTerm = linearTerm;
    this.targetValue = targetValue;
  }

  public SolutionVector(long id, @NotNull SparseVector key, Boolean value, double linearTerm, double alpha) {
    this(id, key, value, linearTerm);
    this.alpha = alpha;
  }

// ------------------------ CANONICAL METHODS ------------------------

  @Override
  public boolean equals(Object o) {
    return o instanceof SolutionVector
        && id == ((SolutionVector) o).id
        && rank == ((SolutionVector) o).rank;
  }

  // PERF hack for speed

  public int hashCode() {
    return Long.hashCode(id);
  }

  @Override
  public String toString() {
    return "SolutionVector{" + "point=" + point + ", targetValue=" + targetValue + ", alpha="
        + alpha
        + ", alphaStatus=" + alphaStatus + ", G=" + G + ", linearTerm=" + linearTerm + ", G_bar="
        + G_bar + '}';
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface Comparable ---------------------

  public int compareTo(SolutionVector b) {
    return Integer.compare(rank, b.rank);
  }

// -------------------------- OTHER METHODS --------------------------

  boolean isFree() {
    return alphaStatus == Status.FREE;
  }

  public boolean isShrinkable(double Gmax1, double Gmax2) {

    if (isUpperBound()) {
      if (targetValue) {
        return -G > Gmax1;
      } else {
        return -G > Gmax2;
      }
    } else if (isLowerBound()) {
      if (targetValue) {
        return G > Gmax2;
      } else {
        return G > Gmax1;
      }
    } else {
      return false;
    }
  }

  protected boolean isUpperBound() {
    return alphaStatus == Status.UPPER_BOUND;
  }

  boolean isLowerBound() {
    return alphaStatus == Status.LOWER_BOUND;
  }

  public boolean isShrinkable(double Gmax1, double Gmax2, double Gmax3, double Gmax4) {
    if (isUpperBound()) {
      if (targetValue) {
        return (-G > Gmax1);
      } else {
        return (-G > Gmax4);
      }
    } else if (isLowerBound()) {
      if (targetValue) {
        return (G > Gmax2);
      } else {
        return (G > Gmax3);
      }
    } else {
      return false;
    }
  }

  public void updateAlphaStatus(double Cp, double Cn) {
    if (alpha >= getC(Cp, Cn)) {
      alphaStatus = Status.UPPER_BOUND;
    } else if (alpha <= 0) {
      alphaStatus = Status.LOWER_BOUND;
    } else {
      alphaStatus = Status.FREE;
    }
  }

  double getC(double Cp, double Cn) {
    return targetValue ? Cp : Cn;
  }

// -------------------------- ENUMERATIONS --------------------------

  public enum Status {
    LOWER_BOUND, UPPER_BOUND, FREE
  }
}
