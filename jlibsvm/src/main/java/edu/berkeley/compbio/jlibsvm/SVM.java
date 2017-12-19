package edu.berkeley.compbio.jlibsvm;

import static com.google.common.base.Preconditions.checkState;

import edu.berkeley.compbio.jlibsvm.crossvalidation.CrossValidationResults;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.stream.Stream;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class SVM<L extends Comparable, R extends SvmProblem<L, R>> implements
    Serializable {

// -------------------------- OTHER METHODS --------------------------

  public Map<SparseVector, Double> continuousCrossValidation(SvmProblem<L, R> problem,
      final ImmutableSvmParameter<L> param) {

    final Map<SparseVector, Double> predictions = new ConcurrentHashMap<>();

    if (param.crossValidationFolds >= problem.getNumExamples()) {
      // this can happen when the points chosen from a multiclass CV don't include enough points from a given pair of classes
      throw new SvmException("Can't have more cross-validation folds than there are examples");
    }

    final Stream<R> foldStream = problem.makeFolds(param.crossValidationFolds);

    foldStream
        .parallel()
        .forEach(fold -> {
          final ContinuousModel model = (ContinuousModel) train(fold, param);

          for (final SparseVector p : fold.getHeldOutPoints()) {
            predictions.put(p, model.predictValue(p));
          }
        });

    // now predictions contains the prediction for each point based on training with e.g. 80% of the other points (for 5-fold).
    return predictions;
  }

  public abstract SolutionModel<L> train(R problem, ImmutableSvmParameter<L> param);

  public Map<SparseVector, L> discreteCrossValidation(SvmProblem<L, R> problem,
      final ImmutableSvmParameter<L> param) {
    final Map<SparseVector, L> predictions = new ConcurrentHashMap<>();
    final Set<SparseVector> nullPredictionPoints =
        new ConcurrentSkipListSet<>(); // necessary because ConcurrentHashMap doesn't support null values

    if (param.crossValidationFolds >= problem.getNumExamples()) {
      throw new SvmException("Can't have more cross-validation folds than there are examples");
    }

    problem
        .makeFolds(param.crossValidationFolds)
        .parallel()
        .forEach(fold -> {
          // this will throw ClassCastException if you try cross-validation on a continuous-only model (e.g. RegressionModel)
          final SolutionModel<L> model = train(fold,
              param);

          checkState(model instanceof DiscreteModel,
              "train() must return an instance of DiscreteModel");

          // note the param has not changed here, so if the method includes oneVsAll models with a
          // probability threshold, those will be independently computed for each fold and so play
          // into the predictLabel

          for (final SparseVector p : fold.getHeldOutPoints()) {
            L prediction = ((DiscreteModel<L>) model).predictLabel(p);
            if (prediction == null) {
              nullPredictionPoints.add(p);
            } else {
              predictions.put(p, prediction);
            }
          }

        });

    // collapse into non-concurrent map that supports null values
    Map<SparseVector, L> result = new HashMap<>(predictions);
    for (SparseVector nullPredictionPoint : nullPredictionPoints) {
      result.put(nullPredictionPoint, null);
    }
    // now predictions contains the prediction for each point based on training with e.g. 80% of the other points (for 5-fold).
    return result;
  }

  public abstract String getSvmType();


  public void validateParam( ImmutableSvmParameter<L> param) {
    if (param.eps < 0) {
      throw new SvmException("eps < 0");
    }
  }

  public abstract CrossValidationResults performCrossValidation(R problem,
      ImmutableSvmParameter<L> param);
}
