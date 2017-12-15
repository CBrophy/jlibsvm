package edu.berkeley.compbio.jlibsvm.multi;

import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterGrid;
import edu.berkeley.compbio.jlibsvm.SVM;
import edu.berkeley.compbio.jlibsvm.binary.BinaryClassificationProblem;
import edu.berkeley.compbio.jlibsvm.binary.BinaryClassificationSVM;
import edu.berkeley.compbio.jlibsvm.binary.BinaryModel;
import edu.berkeley.compbio.jlibsvm.binary.BooleanClassificationProblemImpl;
import edu.berkeley.compbio.jlibsvm.labelinverter.LabelInverter;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.jlibsvm.util.SubtractionMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MultiClassificationSVM<L extends Comparable<L>> extends
    SVM<L, MultiClassProblem<L>> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(MultiClassificationSVM.class);

  private BinaryClassificationSVM<L> binarySvm;

// --------------------------- CONSTRUCTORS ---------------------------

  public MultiClassificationSVM(BinaryClassificationSVM<L> binarySvm) {
    this.binarySvm = binarySvm;
  }

// -------------------------- OTHER METHODS --------------------------

  @Override
  public String getSvmType() {
    return "multiclass " + binarySvm.getSvmType();
  }

  public SvmMultiClassCrossValidationResults<L> performCrossValidation(
      @NotNull MultiClassProblem<L> problem,
      @NotNull ImmutableSvmParameter<L> param) {
    Map<SparseVector, L> predictions = discreteCrossValidation(problem, param);

    SvmMultiClassCrossValidationResults<L> cv =
        new SvmMultiClassCrossValidationResults<L>(problem, predictions);
    cv.param = param;
    return cv;
  }

  public MultiClassModel<L> train(@NotNull MultiClassProblem<L> problem,
      @NotNull ImmutableSvmParameter<L> param) {
    validateParam(param);

    MultiClassModel<L> result;
    if (param instanceof ImmutableSvmParameterGrid
        && !param.gridsearchBinaryMachinesIndependently) {

      // performs cross-validation at each grid point regardless
      result = trainGrid(problem, (ImmutableSvmParameterGrid<L>) param); //, execService);

      //		}
    } else {
      // train once using all the data
      result = trainScaled(problem, param); //, execService);
    }
    return result;
  }

  public MultiClassModel<L> trainGrid(@NotNull final MultiClassProblem<L> problem,
      @NotNull final ImmutableSvmParameterGrid<L> param) {
    final GridTrainingResult gtresult = new GridTrainingResult();

    param
        .getGridParams()
        .stream()
        .parallel()
        .forEach(point -> {
          // note we must use the CV variant in order to know which parameter set is best
          SvmMultiClassCrossValidationResults<L> crossValidationResults =
              performCrossValidation(problem, point); //, execService);
          gtresult.update(crossValidationResults); //
          // if we did a grid search, keep track of which parameter set was used for these results

        });

    logger.info("Chose grid point: " + gtresult.bestCrossValidationResults.param);

    // finally train once on all the data (including rescaling)
    MultiClassModel<L> result =
        trainScaled(problem, gtresult.bestCrossValidationResults.param); //, execService);
    result.crossValidationResults = gtresult.bestCrossValidationResults;
    return result;
  }


  private class GridTrainingResult {

    SvmMultiClassCrossValidationResults<L> bestCrossValidationResults = null;
    double bestSensitivity = -1.0;

    synchronized void update(
        SvmMultiClassCrossValidationResults<L> crossValidationResults) {
      double sensitivity = crossValidationResults.classNormalizedSensitivity();
      if (sensitivity > bestSensitivity) {
        bestSensitivity = sensitivity;
        bestCrossValidationResults = crossValidationResults;
      }
    }
  }


  /**
   * Train the classifier, and also prepare the probability sigmoid thing if requested.
   */

  public MultiClassModel<L> trainScaled(@NotNull final MultiClassProblem<L> problem,
      @NotNull final ImmutableSvmParameter<L> param) {
    if (param.scalingModelLearner != null && !param.scaleBinaryMachinesIndependently) {
      return trainWithoutScaling(problem.getScaledCopy(param.scalingModelLearner),
          param);
    } else {
      return trainWithoutScaling(problem, param);
    }
  }

  private MultiClassModel<L> trainWithoutScaling(@NotNull final MultiClassProblem<L> problem,
      @NotNull final ImmutableSvmParameter<L> param) {
    int numLabels = problem.getLabels().size();

    final MultiClassModel<L> model = new MultiClassModel<>(param, numLabels);

    model.setScalingModel(problem.getScalingModel());

    final Map<L, Set<SparseVector>> examplesByLabel = problem.getExamplesByLabel();

    if (param.oneVsAllMode != MultiClassModel.OneVsAllMode.None) {
      // create and train one vs all classifiers.

      // oneVsAll models always need a probability sigmoid

      final ImmutableSvmParameter<L> probParam = param.withProbabilityCopy();

      // first queue up all the training tasks and submit them to the thread pool

      logger.info("Training one-vs-all classifiers for " + numLabels + " labels");

      final LabelInverter<L> labelInverter = problem.getLabelInverter();

      problem
          .getLabels()
          .parallelStream()
          .forEach(label -> {
            final L notLabel = labelInverter.invert(label);

            final Set<SparseVector> labelExamples = examplesByLabel.get(label);

            Collection<Map.Entry<SparseVector, L>> entries = problem.getExamples().entrySet();
            if (param.falseClassSVlimit != Integer.MAX_VALUE) {
              // guarantee entries in random order if limiting the number of false examples
              List<Map.Entry<SparseVector, L>> entryList = new ArrayList<>(entries);
              Collections.shuffle(entryList);
              int toIndex = param.falseClassSVlimit + labelExamples.size();
              toIndex = Math.min(toIndex, entryList.size());
              entries = entryList.subList(0, toIndex);
            }

            final Set<SparseVector> notlabelExamples =
                new SubtractionMap<>(entries, labelExamples, param.falseClassSVlimit).keySet();

            final BinaryClassificationProblem<L> subProblem =
                new BooleanClassificationProblemImpl<>(problem.getLabelClass(), label,
                    labelExamples,
                    notLabel, notlabelExamples);
            // Unbalanced data: see prepareWeights
            // since these will be extremely unbalanced, this should nearly guarantee that no positive examples are misclassified.

            BinaryModel<L> result = binarySvm.train(subProblem, probParam); //, execService);
            model.putOneVsAllModel(label, result);
          });


    }

    if (param.allVsAllMode != MultiClassModel.AllVsAllMode.None) {
      final int numClassifiers = (numLabels * (numLabels - 1)) / 2;
      // create and train all vs all classifiers

      // first queue up all the training tasks and submit them to the thread pool

      logger.info(
          "Training " + numClassifiers + " one-vs-one classifiers for " + numLabels + " labels");
      int c = 0;

      problem
          .getLabels()
          .stream()
          .flatMap(label ->
              problem
                  .getLabels()
                  .stream()
                  .map(label2 -> new Object[]{label, label2})
          )
          .parallel()
          .forEach(pair -> {
            final L label1 = (L) pair[0];
            final L label2 = (L) pair[1];

            final Set<SparseVector> label1Examples = examplesByLabel.get(label1);
            final Set<SparseVector> label2Examples = examplesByLabel.get(label2);

            final BinaryClassificationProblem<L> subProblem =
                new BooleanClassificationProblemImpl<>(problem.getLabelClass(), label1,
                    label1Examples,
                    label2, label2Examples);

            BinaryModel<L> result = binarySvm.train(subProblem, param);
            model.putOneVsOneModel(label1, label2, result);
          });

    }

    model.prepareModelSvMaps();
    return model;
  }

}
