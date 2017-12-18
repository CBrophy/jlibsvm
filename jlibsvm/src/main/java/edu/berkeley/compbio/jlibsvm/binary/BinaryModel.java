package edu.berkeley.compbio.jlibsvm.binary;

import edu.berkeley.compbio.jlibsvm.ContinuousModel;
import edu.berkeley.compbio.jlibsvm.DiscreteModel;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameterPoint;
import edu.berkeley.compbio.jlibsvm.LabelParser;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;
import edu.berkeley.compbio.jlibsvm.scaler.NoopScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.Serializable;
import java.util.Collection;
import java.util.Properties;
import java.util.StringTokenizer;
import java.util.logging.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class BinaryModel<L extends Comparable> extends AlphaModel<L>
    implements DiscreteModel<L>, ContinuousModel, Serializable {
  // protected final would be nice, but the Solver constructs the Model without knowing about param so we have to set it afterwards.
  /**
   * a thing that is confusing here: if a grid search was done, then the specific point that was the optimum should be
   * recorded here.  That works for binary and multiclass models when the grid search is done at the top level.  But when
   * param.gridSearchBinaryMachinesIndependently, there is no one point that makes sense.  Really we should just leave it
   * null and refer to the subsidiary BinaryModels.
   */
  public ImmutableSvmParameterPoint<L> param;

// ------------------------------ FIELDS ------------------------------

  public double obj;
  public double upperBoundPositive;
  public double upperBoundNegative;

  public ScalingModel scalingModel = new NoopScalingModel();

  public double r;// for Solver_NU.  I wanted to factor this out as SolutionInfoNu, but that was too much hassle
  public SvmBinaryCrossValidationResults<L> crossValidationResults;

  public SvmBinaryCrossValidationResults<L> getCrossValidationResults() {
    return crossValidationResults;
  }

  L trueLabel;
  L falseLabel;


  public Collection<L> getLabels() {
    return param.getLabels();
  }

  @Override
  public String getKernelName() {
    return param.kernel.toString();
  }

  // --------------------------- CONSTRUCTORS ---------------------------

  public BinaryModel() {
    super();
  }


  public BinaryModel(Properties props, LabelParser<L> labelParser) {

    ImmutableSvmParameterPoint.Builder<L> builder = new ImmutableSvmParameterPoint.Builder<L>();
    try {
      builder.kernel =
          (KernelFunction) Class.forName(props.getProperty("kernel_type"))
              .getConstructor(Properties.class)
              .newInstance(props);
    } catch (Throwable e) {
      throw new SvmException(e);
    }
    StringTokenizer st = new StringTokenizer(props.getProperty("label"));
    while (st.hasMoreTokens()) {
      builder.putWeight(labelParser.parse(st.nextToken()), null);
    }

    rho = Double.parseDouble(props.getProperty("rho"));
    numSVs = Integer.parseInt(props.getProperty("total_sv"));
//** test hack

    trueLabel = (L) "true";
    falseLabel = (L) "false";
    param = builder.build();
  }

  public BinaryModel(ImmutableSvmParameterPoint<L> param) {
    this.param = param;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  public L getFalseLabel() {
    return falseLabel;
  }

  @NotNull
  public ScalingModel getScalingModel() {
    return scalingModel;
  }

  public void setScalingModel(@NotNull ScalingModel scalingModel) {
    this.scalingModel = scalingModel;
  }

  public L getTrueLabel() {
    return trueLabel;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface DiscreteModel ---------------------

  public L predictLabel(SparseVector x) {
    return predictValue(x) > 0 ? trueLabel : falseLabel;
  }

// -------------------------- OTHER METHODS --------------------------

  public double getSumAlpha() {
    double result = 0;
    for (Double aFloat : supportVectors.values()) {
      result += aFloat;
    }
    return result;
  }

  public double getTrueProbability(SparseVector x) {
    return crossValidationResults.sigmoid.predict(predictValue(x));  // NPE if no sigmoid
  }

  public double getProbability(SparseVector x, L l) {
    if (l.equals(trueLabel)) {
      return getTrueProbability(x);
    } else if (l.equals(falseLabel)) {
      return 1.0 - getTrueProbability(x);
    } else {
      throw new SvmException(
          "Can't compute probability: " + l + " is not one of the classes in this binary model ("
              + trueLabel
              + ", " + falseLabel + ")");
    }
  }

  public double predictValue(SparseVector x) {
    double sum = 0;

    SparseVector scaledX = scalingModel.scaledCopy(x);

    for (int i = 0; i < numSVs; i++) {
      double kvalue = param.kernel.evaluate(scaledX, SVs[i]);
      sum += alphas[i] * kvalue;
    }

    sum -= rho;
    return sum;
  }

  public double getTrueProbability(double[] kvalues, int[] svIndexMap) {
    double pv = predictValue(kvalues, svIndexMap);
    if (crossValidationResults == null) {
      Logger.getGlobal()
          .severe("Can't compute probability in binary model without crossvalidationresults");
      return pv > 0. ? 1.0 : 0.0;
    } else if (crossValidationResults.sigmoid == null) {
      Logger.getGlobal().severe("Can't compute probability in binary model without sigmoid");
      return pv > 0. ? 1.0 : 0.0;
    } else {
      return crossValidationResults.sigmoid.predict(pv);  // NPE if no sigmoid
    }
  }

  public double predictValue(double[] kvalues, int[] svIndexMap) {
    double sum = 0;

    for (int i = 0; i < numSVs; i++) {
      sum += alphas[i] * kvalues[svIndexMap[i]];
    }

    sum -= rho;
    return sum;
  }

  public L predictLabel(double[] kvalues, int[] svIndexMap) {
    return predictValue(kvalues, svIndexMap) > 0 ? trueLabel : falseLabel;
  }

}
