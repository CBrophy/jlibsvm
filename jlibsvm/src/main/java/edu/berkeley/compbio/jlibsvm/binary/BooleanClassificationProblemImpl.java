package edu.berkeley.compbio.jlibsvm.binary;

import com.google.common.collect.HashMultiset;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModelLearner;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BooleanClassificationProblemImpl<L extends Comparable> extends
    BinaryClassificationProblemImpl<L> {
// ------------------------------ FIELDS ------------------------------

  private Map<SparseVector, Boolean> booleanExamples;
  private Set<SparseVector> trueExamples;
  private Set<SparseVector> falseExamples;
  int numExamples = 0;

// --------------------------- CONSTRUCTORS ---------------------------

  public BooleanClassificationProblemImpl(Class labelClass, L trueLabel,
      Set<SparseVector> trueExamples,
      L falseLabel,
      Set<SparseVector> falseExamples) {
    // this is a hack: we leave examples==null and just deal with booleanExamples directly

    super(labelClass, null);
    this.falseLabel = falseLabel;
    this.trueLabel = trueLabel;
    this.trueExamples = trueExamples;
    this.falseExamples = falseExamples;

    labels = new ArrayList<L>(2);
    labels.add(trueLabel);
    labels.add(falseLabel);

    numExamples = trueExamples.size() + falseExamples.size();

    exampleCounts = HashMultiset.create();
    exampleCounts.add(trueLabel, trueExamples.size());
    exampleCounts.add(falseLabel, falseExamples.size());
  }


  public BooleanClassificationProblemImpl(BooleanClassificationProblemImpl<L> backingProblem,
      Set<SparseVector> heldOutPoints) {
    super(backingProblem.labelClass, null, backingProblem.scalingModel,
        backingProblem.trueLabel, backingProblem.falseLabel);
    this.heldOutPoints = heldOutPoints;

    // PERF use a SubtractionSet?
    this.trueExamples = new HashSet<>(backingProblem.trueExamples);
    this.falseExamples = new HashSet<>(backingProblem.falseExamples);
    trueExamples.removeAll(heldOutPoints);
    falseExamples.removeAll(heldOutPoints);

    labels = new ArrayList<>(2);
    labels.add(trueLabel);
    labels.add(falseLabel);

    numExamples = trueExamples.size() + falseExamples.size();

    exampleCounts = HashMultiset.create();
    exampleCounts.add(trueLabel, trueExamples.size());
    exampleCounts.add(falseLabel, falseExamples.size());
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public synchronized Map<SparseVector, Boolean> getBooleanExamples() {
    if (booleanExamples == null) {
      booleanExamples = new HashMap<>(numExamples);
      for (SparseVector trueExample : trueExamples) {
        booleanExamples.put(trueExample, Boolean.TRUE);
      }
      for (SparseVector falseExample : falseExamples) {
        booleanExamples.put(falseExample, Boolean.FALSE);
      }
      assert booleanExamples.size() == numExamples;
    }
    return booleanExamples;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface BinaryClassificationProblem ---------------------

  /**
   * There's no sense in scaling Boolean values, so this is a noop.  note we don't make a copy for efficiency.
   */
  public BinaryClassificationProblem<L> getScaledCopy(
       ScalingModelLearner scalingModelLearner) {
    return this;
  }

  public void setupLabels() {
    // the constructor already dealt with this
  }

// --------------------- Interface SvmProblem ---------------------

  public synchronized L getTargetValue(SparseVector point) {
    if (booleanExamples.get(point)) {
      return trueLabel;
    } else {
      return falseLabel;
    }
  }

  public int getNumExamples() {
    return numExamples;
  }

  // need to override this because of the examples == null hack
  public Stream<BinaryClassificationProblem<L>> makeFolds(int numberOfFolds) {
//		Set<BinaryClassificationProblem<L, P>> result = new HashSet<BinaryClassificationProblem<L, P>>();

    List<SparseVector> points = new ArrayList<>(getBooleanExamples().keySet());

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

  }


  protected BooleanClassificationProblemImpl<L> makeFold(Set<SparseVector> heldOutPoints) {
    return new BooleanClassificationProblemImpl(this, heldOutPoints);
  }
}
