package edu.berkeley.compbio.jlibsvm.binary;

import com.google.common.base.Throwables;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.SVM;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import java.util.Map;
import java.util.logging.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class BinaryClassificationSVM<L extends Comparable>
    extends SVM<L, BinaryClassificationProblem<L>> implements Serializable {

// -------------------------- OTHER METHODS --------------------------


  public BinaryModel<L> train(@NotNull BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameter<L> param)
  {
    validateParam(param);
    BinaryModel<L> result;
    if (param instanceof ImmutableSvmParameterGrid)  //  either the problem was binary to start with, or param.gridSearchBinaryMachinesIndependently
    {
      result = trainGrid(problem, (ImmutableSvmParameterGrid<L>) param);
    } else if (param.probability)  // this may already be a fold, but we have to sub-fold it to get probabilities
    {
      result = trainScaledWithCV(problem,
          (ImmutableSvmParameterPoint<L>) param);
    } else {
      result = trainScaled(problem, (ImmutableSvmParameterPoint<L>) param);
    }
    return result;
  }

  /**
   * Try a bunch of different parameter sets, and return the model based on the one that produces the best
   * class-normalized sensitivity.
   */


  private BinaryModel<L> trainGrid(@NotNull final BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameterGrid<L> param)
  {
    final GridTrainingResult gtresult = new GridTrainingResult();

    param
        .getGridParams()
        .parallelStream()
        .forEach(point -> {
          // note we must use the CV variant in order to know which parameter set is best
          SvmBinaryCrossValidationResults<L> crossValidationResults =
              performCrossValidation(problem, point);
          Logger.getGlobal().info("CV results for grid point " + point + ": " + crossValidationResults);
          gtresult.update(point, crossValidationResults);

        });

    // no need for the iterator version here; the set of params doesn't require too much memory
    Logger.getGlobal().info("Chose grid point: " + gtresult.bestParam);

    // finally train once on all the data (including rescaling)
    BinaryModel<L> result = trainScaled(problem, gtresult.bestParam);
    synchronized (gtresult) {
      result.crossValidationResults = gtresult.bestCrossValidationResults;
    }
    return result;
  }

  private class GridTrainingResult {

    ImmutableSvmParameterPoint<L> bestParam = null;
    SvmBinaryCrossValidationResults<L> bestCrossValidationResults = null;
    double bestSensitivity = -1F;

    synchronized void update(ImmutableSvmParameterPoint<L> gridParam,
        SvmBinaryCrossValidationResults<L> crossValidationResults) {
      double sensitivity = crossValidationResults.classNormalizedSensitivity();
      if (sensitivity > bestSensitivity) {
        bestParam = gridParam;
        bestSensitivity = sensitivity;
        bestCrossValidationResults = crossValidationResults;
      }
    }
  }

  /**
   * Train the classifier, and also prepare the probability sigmoid thing if requested.
   */
  private BinaryModel<L> trainScaledWithCV(@NotNull BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameterPoint<L> param)
  {
    // if scaling each binary machine is enabled, then each fold will be independently scaled also; so we don't need to scale the whole dataset prior to CV

    SvmBinaryCrossValidationResults<L> cv = null;
    try {
      cv = performCrossValidation(problem, param);
    } catch (SvmException e) {
      //ignore, probably there weren't enough points to make folds
      Logger.getGlobal().info("Could not perform cross-validation\n" + Throwables.getStackTraceAsString(e));
    }

    // finally train once on all the data (including rescaling)
    BinaryModel<L> result = trainScaled(problem, param);
    result.crossValidationResults = cv;  // careful later: this might be null

    return result;
  }


  public SvmBinaryCrossValidationResults<L> performCrossValidation(
      @NotNull BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameter<L> param)
  {
    //there is no point in computing probabilities on these submodels (and that produces infinite recursion)
    ImmutableSvmParameterPoint<L> noProbParam = (ImmutableSvmParameterPoint<L>) param
        .noProbabilityCopy();

    final Map<SparseVector, Double> decisionValues = continuousCrossValidation(problem,
        noProbParam);

    // but the CV may be used to compute probabilities at this level, if requested
    SvmBinaryCrossValidationResults<L> cv =
        new SvmBinaryCrossValidationResults<L>(problem, decisionValues, param.probability);
    return cv;
  }

  /**
   * Normal training on the entire problem, with no scaling and no cross-validation-based probability measure.
   */
  protected abstract BinaryModel<L> trainOne(@NotNull BinaryClassificationProblem<L> problem,
      double Cp,
      double Cn, @NotNull ImmutableSvmParameterPoint<L> param);


  private BinaryModel<L> trainScaled(@NotNull BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameterPoint<L> param) {
    if (param.scalingModelLearner != null && param.scaleBinaryMachinesIndependently) {
      // the examples are copied before scaling, not scaled in place
      // that way we don't need to worry that the same examples are being used in another thread, or scaled differently in different contexts, etc.
      // this may cause memory problems though

      problem = problem.getScaledCopy(param.scalingModelLearner);
    }

    BinaryModel<L> result = trainWeighted(problem, param);

    return result;
  }

  private BinaryModel<L> trainWeighted(@NotNull BinaryClassificationProblem<L> problem,
      @NotNull ImmutableSvmParameterPoint<L> param) {
    // calculate weighted C

    double weightedCp = param.C;
    double weightedCn = param.C;

    if (param.redistributeUnbalancedC) {
      Double weightP = param.getWeight(problem.getTrueLabel());
      if (weightP != null) {
        weightedCp *= weightP;
      }

      Double weightN = param.getWeight(problem.getFalseLabel());
      if (weightN != null) {
        weightedCn *= weightN;
      }
    }

    // train using those
    BinaryModel<L> result = trainOne(problem, weightedCp, weightedCn, param);

    return result;
  }

}
