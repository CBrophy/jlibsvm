package edu.berkeley.compbio.jlibsvm;

import edu.berkeley.compbio.jlibsvm.scaler.NoopScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class ExplicitSvmProblemImpl<L extends Comparable, R extends SvmProblem<L, R>>
    extends AbstractSvmProblem<L, R> implements ExplicitSvmProblem<L, R> {
// ------------------------------ FIELDS ------------------------------

  public Map<SparseVector, L> examples;

  public ScalingModel scalingModel = new NoopScalingModel();

  /**
   * the unique set of targetvalues, in a defined order avoid populating for regression!  OK, regression should never
   * call getLabels(), then.
   */
  protected List<L> labels = null;

// --------------------------- CONSTRUCTORS ---------------------------

  protected ExplicitSvmProblemImpl(Map<SparseVector, L> examples) {
    this.examples = examples;
  }

  protected ExplicitSvmProblemImpl(Map<SparseVector, L> examples,
      @NotNull ScalingModel scalingModel) {
    this.examples = examples;
    this.scalingModel = scalingModel;
  }

  protected ExplicitSvmProblemImpl(Map<SparseVector, L> examples,
      @NotNull ScalingModel scalingModel, Set<SparseVector> heldOutPoints) {
    this.examples = examples;
    this.scalingModel = scalingModel;
    this.heldOutPoints = heldOutPoints;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  @NotNull
  public Map<SparseVector, L> getExamples() {
    return examples;
  }

  public List<L> getLabels() {
    if (labels == null) {
      if (examples.isEmpty()) {
        return null;
      }
      Set<L> uniq = new HashSet<>(examples.values());
      labels = new ArrayList<>(uniq);
      Collections.sort(labels);
    }
    return labels;
  }

  @NotNull
  public ScalingModel getScalingModel() {
    return scalingModel;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface ExplicitSvmProblem ---------------------

  public Stream<R> makeFolds(int numberOfFolds) {

    List<SparseVector> points = new ArrayList<>(getExamples().keySet());

    Collections.shuffle(points);

    // PERF this is maybe overwrought, but ensures the best possible balance among folds (unlike examples.size() / numberOfFolds)

    List<Set<SparseVector>> heldOutPointSets = new ArrayList<>();
    for (int i = 0; i < numberOfFolds; i++) {
      heldOutPointSets.add(new HashSet<>());
    }

    int f = 0;
    for (SparseVector point : points) {
      heldOutPointSets.get(f).add(point);
      f++;
      f %= numberOfFolds;
    }

    return heldOutPointSets
        .stream()
        .map(this::makeFold);
//
//    Iterator<R> foldIterator = new MappingIterator<Set<SparseVector>, R>(heldOutPointSets.iterator()) {
//      @NotNull
//      public R function(Set<SparseVector> p) {
//        return makeFold(p);
//      }
//    };
//    return foldIterator;

  }

// --------------------- Interface SvmProblem ---------------------

  public L getTargetValue(SparseVector point) {
    return examples.get(point);
  }

  public int getNumExamples() {

    return examples.size();
  }

  // -------------------------- OTHER METHODS --------------------------

  protected Set<SparseVector> heldOutPoints = new HashSet<>();


  public Set<SparseVector> getHeldOutPoints() {
    return heldOutPoints;
  }


  protected abstract R makeFold(Set<SparseVector> heldOutPoints);
}
