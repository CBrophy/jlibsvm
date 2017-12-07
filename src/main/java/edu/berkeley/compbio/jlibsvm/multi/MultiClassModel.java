package edu.berkeley.compbio.jlibsvm.multi;

import com.google.common.base.Function;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Multiset;
import edu.berkeley.compbio.jlibsvm.DiscreteModel;
import edu.berkeley.compbio.jlibsvm.ImmutableSvmParameter;
import edu.berkeley.compbio.jlibsvm.SolutionModel;
import edu.berkeley.compbio.jlibsvm.SvmException;
import edu.berkeley.compbio.jlibsvm.binary.BinaryModel;
import edu.berkeley.compbio.jlibsvm.kernel.KernelFunction;
import edu.berkeley.compbio.jlibsvm.scaler.NoopScalingModel;
import edu.berkeley.compbio.jlibsvm.scaler.ScalingModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import edu.berkeley.compbio.ml.MultiClassCrossValidationResults;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.log4j.Logger;
import org.jetbrains.annotations.NotNull;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class MultiClassModel<L extends Comparable> extends SolutionModel<L> implements
    DiscreteModel<L> {
// ------------------------------ FIELDS ------------------------------

  private static final Logger logger = Logger.getLogger(MultiClassModel.class);

  private ScalingModel scalingModel = new NoopScalingModel();

  private final OneVsAllMode oneVsAllMode;
  private final double oneVsAllThreshold;

  private final AllVsAllMode allVsAllMode;
  private final double minVoteProportion;

  private final Map<BinaryModel<L>, int[]> svIndexMaps; // = new HashMap<BinaryModel<L, P>, int[]>();

  private final int numberOfClasses;

  private final SymmetricHashMap2d<L, BinaryModel<L>> oneVsOneModels;
  private final HashMap<L, BinaryModel<L>> oneVsAllModels;

  private SparseVector[] allSVs;
  SvmMultiClassCrossValidationResults<L> crossValidationResults;

  public MultiClassCrossValidationResults<L> getCrossValidationResults() {
    return crossValidationResults;
  }

  /**
   * Make a derived copy for leave-one-out testing
   */
  public MultiClassModel(MultiClassModel<L> copyFrom, Collection<L> excludeLabels) {

    allSVs = copyFrom.allSVs;  // the labels list provides the indexes for this array, which we also don't change

    oneVsAllMode = copyFrom.oneVsAllMode;
    oneVsAllThreshold = copyFrom.oneVsAllThreshold;
    allVsAllMode = copyFrom.allVsAllMode;
    minVoteProportion = copyFrom.minVoteProportion;
    numberOfClasses = copyFrom.numberOfClasses;
    svIndexMaps = copyFrom.svIndexMaps;

    scalingModel = copyFrom.scalingModel;

    // the only thing that does change is that some binary models are excluded

    oneVsOneModels = new SymmetricHashMap2d<>(copyFrom.oneVsOneModels,
        excludeLabels);
    oneVsAllModels = new HashMap<>(copyFrom.oneVsAllModels);
    for (L disallowedLabel : excludeLabels) {
      oneVsAllModels.remove(disallowedLabel);
    }
  }

// --------------------------- CONSTRUCTORS ---------------------------

  public MultiClassModel(ImmutableSvmParameter param, int numberOfClasses) {
    //super(param);
    super();
    svIndexMaps = new HashMap<>();
    this.numberOfClasses = numberOfClasses;
    oneVsOneModels = new SymmetricHashMap2d<>(numberOfClasses);
    oneVsAllModels = new HashMap<>(numberOfClasses);

    this.oneVsAllThreshold = param.oneVsAllThreshold;

    this.oneVsAllMode = param.oneVsAllMode;
    this.allVsAllMode = param.allVsAllMode;
    this.minVoteProportion = param.minVoteProportion;
  }

// --------------------- GETTER / SETTER METHODS ---------------------

  @NotNull
  public ScalingModel getScalingModel() {
    return scalingModel;
  }

  public void setScalingModel(@NotNull ScalingModel scalingModel) {
    this.scalingModel = scalingModel;
  }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface DiscreteModel ---------------------

  /**
   * @return null if no good label is found, otherwise the best label.
   */
  public L predictLabel(SparseVector x) {
    return predictLabelWithQuality(x).getBestLabel();
  }

// -------------------------- OTHER METHODS --------------------------

  public L bestProbabilityLabel(Map<L, Double> labelProbabilities) {
    Double bestProb = 0.0;
    L bestLabel = null;
    for (Map.Entry<L, Double> entry : labelProbabilities.entrySet()) {
      if (entry.getValue() > bestProb) {
        bestLabel = entry.getKey();
        bestProb = entry.getValue();
      }
    }
    return bestLabel;
  }

  @NotNull
  public VotingResult<L> predictLabelWithQuality(SparseVector x)
  {
    final SparseVector scaledX = scalingModel.scaledCopy(x);

    L bestLabel = null;

    double bestOneClassProbability = 0;
    double secondBestOneClassProbability = 0;

    double bestOneVsAllProbability = 0;
    double secondBestOneVsAllProbability = 0;

    // stage 0: we're going to need the kernel value for x against each of the SVs, for each of the kernels that was used in a subsidary binary machine

    Map<KernelFunction, double[]> kValuesPerKernel =
        new MapMaker().makeComputingMap(new Function<KernelFunction, double[]>() {
          public double[] apply(@NotNull KernelFunction kernel) {
            double[] kvalues = new double[allSVs.length];
            int i = 0;
            for (SparseVector sv : allSVs) {
              kvalues[i] = kernel.evaluate(scaledX, sv);
              i++;
            }
            return kvalues;
          }
        });

    // we don't want to consider any models that mention a disallowed label
    // (i.e., not only should such a prediction be rejected after the fact, but
    //  the binary machines involving disallowed labels shouldn't ever contribute to the voting in the first place

    // stage 2: one vs all

    Map<L, Double> oneVsAllProbabilities =
        oneVsAllMode == OneVsAllMode.None ? null : computeOneVsAllProbabilities(kValuesPerKernel);

    // now oneVsAllProbabilities is populated with all of the classes that pass the threshold (maybe all of them).

    // if all classes were vetoed, return
    if ((oneVsAllMode == OneVsAllMode.Veto || oneVsAllMode == OneVsAllMode.VetoAndBreakTies
        || oneVsAllMode == OneVsAllMode.Best) && oneVsAllProbabilities.isEmpty()) {
      return new VotingResult<L>();
    }

    // if using the OneVsAll Best mode, then we should have had probabilities turned on, and allVsAll voting will be ignored
    if (oneVsAllMode == OneVsAllMode.Best) {
      for (Map.Entry<L, Double> entry : oneVsAllProbabilities.entrySet()) {
        if (entry.getValue() > bestOneVsAllProbability) {
          secondBestOneVsAllProbability = bestOneVsAllProbability;
          bestLabel = entry.getKey();
          bestOneVsAllProbability = entry.getValue();
        }
      }
      return new VotingResult<L>(bestLabel, 0, 0, bestOneClassProbability,
          secondBestOneClassProbability,
          bestOneVsAllProbability, secondBestOneVsAllProbability);
    }

    // stage 3: voting

    int numLabels = oneVsOneModels.keySet().size();

    Multiset<L> votes = HashMultiset.create();

    if (allVsAllMode == AllVsAllMode.AllVsAll) {
      // vote using all models

      logger.debug("Sample voting using all pairs of " + numLabels + " labels ("
          + ((numLabels * (numLabels - 1)) / 2. - numLabels) + " models)");

      // How AllVsAll with Veto differs from FilteredVsAll, etc.:
      // In the AllVsAll with Veto case, we may compute votes between two "inactive" (vetoed) classes;
      // it may be that the winner of the voting later fails the oneVsAll filter, in which
      // case we may want to report unknown instead of reporting the best class that does pass.
      // This is what PhyloPythia does.

      for (BinaryModel<L> binaryModel : oneVsOneModels.values()) {
        double[] kvalues = kValuesPerKernel.get(binaryModel.param.kernel);
        votes.add(binaryModel.predictLabel(kvalues, svIndexMaps.get(binaryModel)));
      }
    } else {
      //vote using only the active models one one side of the comparison, maybe on both.

      Set<L> activeClasses =
          oneVsAllProbabilities != null ? oneVsAllProbabilities.keySet() : oneVsOneModels.keySet();

      int requiredActive = allVsAllMode == AllVsAllMode.FilteredVsAll ? 1 : 2;

      int numActive = oneVsAllProbabilities != null ? oneVsAllProbabilities.size() : numLabels;
      if (requiredActive == 1) {
        logger
            .debug("Sample voting with all " + numLabels + " vs. " + numActive + " active labels ("
                + ((numLabels * (numActive - 1)) / 2. - numActive) + " models)");
      } else {
        logger.debug("Sample voting using pairs of only " + numActive + " active labels ("
            + ((numActive * (numActive - 1)) / 2. - numActive) + " models)");
      }

      for (BinaryModel<L> binaryModel : oneVsOneModels.values()) {
        int activeCount = (activeClasses.contains(binaryModel.getTrueLabel()) ? 1 : 0) + (
            activeClasses.contains(binaryModel.getFalseLabel()) ? 1 : 0);

        if (activeCount >= requiredActive) {
          votes.add(binaryModel.predictLabel(scaledX));
        }
      }
    }

    // stage 4: find the label with the most votes (and break ties or veto as needed)

    int bestCount = 0;
    int secondBestCount = 0;

    int countSum = 0;
    for (L label : votes.elementSet()) {
      int count = votes.count(label);
      countSum += count;

      // get the oneVsAll value for this label, if needed
      Double oneVsAll = 1.0; // pass by default
      if (oneVsAllMode == OneVsAllMode.Veto || oneVsAllMode == OneVsAllMode.VetoAndBreakTies) {
        // if this is null it means this label didn't pass the threshold earlier, so it should fail here too
        oneVsAll = oneVsAllProbabilities.get(label);
        oneVsAll = oneVsAll == null ? 0.0 : oneVsAll;
      }

      // get the oneClass value for this label, if needed

      // if this is null it means this label didn't pass the threshold earlier
      //	Double oneClass = oneClassProbabilities.get(label);
      //	oneClass = oneClass == null ? 0f : oneClass;

      // primary sort by number of votes
      // secondary sort by one-vs-all probability, if available
      // tertiary sort by one-class probability, if available

      if (count > bestCount || (count == bestCount && oneVsAll > bestOneVsAllProbability))
      {
        secondBestCount = bestCount;
        secondBestOneVsAllProbability = bestOneVsAllProbability;

        bestLabel = label;
        bestCount = count;
        bestOneVsAllProbability = oneVsAll;
      }
    }

    // stage 5: check for inadequate evidence filters.

    double bestVoteProportion = (double) bestCount / (double) countSum;
    double secondBestVoteProportion = (double) secondBestCount / (double) countSum;
    if (bestVoteProportion < minVoteProportion) {
      return new VotingResult<>();
    }

    if ((oneVsAllMode == OneVsAllMode.VetoAndBreakTies || oneVsAllMode == OneVsAllMode.Veto)
        && bestOneVsAllProbability < oneVsAllThreshold) {
      return new VotingResult<>();
    }

    return new VotingResult<>(bestLabel, bestVoteProportion,
        secondBestVoteProportion,
        bestOneClassProbability, secondBestOneClassProbability, bestOneVsAllProbability,
        secondBestOneVsAllProbability);
  }


  public Map<L, Double> computeOneVsAllProbabilities(
      Map<KernelFunction, double[]> kValuesPerKernel) {
    Map<L, Double> oneVsAllProbabilities = new HashMap<L, Double>();

    for (BinaryModel<L> binaryModel : oneVsAllModels.values()) {
      double[] kvalues = kValuesPerKernel.get(binaryModel.param.kernel);

      // if probability info isn't available, just substitute 1 and 0.
      final double probability = binaryModel
          .getTrueProbability(kvalues, svIndexMaps.get(binaryModel));

      if (probability >= oneVsAllThreshold) {
        oneVsAllProbabilities.put(binaryModel.getTrueLabel(), probability);
      }
    }
    return oneVsAllProbabilities;
  }

  public Map<L, Double> predictProbability(SparseVector x) {
    if (!supportsOneVsOneProbability()) {
      throw new SvmException("Can't make probability predictions");
    }

    //** ugly Map2d vs. array issue etc.; oh well, adapt for now to the old multiclassProbability signature
    // the main thing is just to iterate through the Map2d in the order given by the labels list

    double minimumProbability = 1e-7;
    double[][] pairwiseProbabilities = new double[numberOfClasses][numberOfClasses];

    // this is kind of a lame way to do it, but whatever.

    // label of each class, just to maintain a known order for the sake of keeping the decision_values etc. straight
    //** proscribed 1-D order for 2-D decision_values is error-prone

    List<L> labels = new ArrayList<>(oneVsOneModels.keySet());

    assert labels.size() == numberOfClasses;

    for (int i = 0; i < numberOfClasses; i++) {
      L label1 = labels.get(i);
      for (int j = i + 1; j < numberOfClasses; j++) {
        L label2 = labels.get(j);

        BinaryModel<L> binaryModel = oneVsOneModels.get(label1, label2);

        if (binaryModel == null) {
          // leave-one-out forbids use of this model, so probability = 0
          pairwiseProbabilities[i][j] = 0;
          pairwiseProbabilities[j][i] = 0;
        } else {
          double prob = binaryModel.crossValidationResults.getSigmoid()
              .predict(binaryModel.predictValue(
                  x));

          pairwiseProbabilities[i][j] = Math
              .min(Math.max(prob, minimumProbability), 1 - minimumProbability);
          pairwiseProbabilities[j][i] = 1 - pairwiseProbabilities[i][j];
        }
      }
    }
    double[] probabilityEstimates = multiclassProbability(numberOfClasses, pairwiseProbabilities);

    // but then map back to the cleaner Map API.  Note the probabilityEstimates should come back in order corresponding to the labels list.

    Map<L, Double> result = new HashMap<L, Double>();
    int i = 0;
    for (L label : labels) {
      result.put(label, probabilityEstimates[i]);
      i++;
    }

    return result;
  }

  public boolean supportsOneVsOneProbability() {
    // just check the first model and assume the rest are the same
    return oneVsOneModels.valueIterator().next().crossValidationResults
        != null;//		return probA != null && probB != null;
  }

  // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng

  private double[] multiclassProbability(int k, double[][] r) {
    double[] p = new double[k];
    int t, j;
    int iter = 0, maximumIterations = Math.max(100, k);
    double[][] Q = new double[k][k];
    double[] Qp = new double[k];
    double pQp, eps = 0.005 / k;

    for (t = 0; t < k; t++) {
      p[t] = 1.0 / k;// Valid if k = 1
      Q[t][t] = 0;
      for (j = 0; j < t; j++) {
        Q[t][t] += r[j][t] * r[j][t];
        Q[t][j] = Q[j][t];
      }
      for (j = t + 1; j < k; j++) {
        Q[t][t] += r[j][t] * r[j][t];
        Q[t][j] = -r[j][t] * r[t][j];
      }
    }
    for (iter = 0; iter < maximumIterations; iter++) {
      // stopping condition, recalculate QP,pQP for numerical accuracy
      pQp = 0;
      for (t = 0; t < k; t++) {
        Qp[t] = 0;
        for (j = 0; j < k; j++) {
          Qp[t] += Q[t][j] * p[j];
        }
        pQp += p[t] * Qp[t];
      }
      double maxError = 0;
      for (t = 0; t < k; t++) {
        double error = Math.abs(Qp[t] - pQp);
        if (error > maxError) {
          maxError = error;
        }
      }
      if (maxError < eps) {
        break;
      }

      for (t = 0; t < k; t++) {
        double diff = (-Qp[t] + pQp) / Q[t][t];
        p[t] += diff;
        pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
        for (j = 0; j < k; j++) {
          Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
          p[j] /= (1 + diff);
        }
      }
    }
    if (iter >= maximumIterations) {
      logger.error("Multiclass probability attempted too many iterations");
    }
    return p;
  }

  public void prepareModelSvMaps() {
    int totalSVs = 0;
    Map<SparseVector, Integer> allSVsMap = new HashMap<>();
    for (BinaryModel<L> binaryModel : oneVsAllModels.values()) {
      int[] svIndexMap = new int[binaryModel.SVs.length];
      int i = 0;
      for (SparseVector p : binaryModel.SVs) {
        Integer allSVsIndex = allSVsMap.get(p);
        if (allSVsIndex == null) {
          allSVsIndex = totalSVs;
          allSVsMap.put(p, allSVsIndex);
          totalSVs++;
        }
        svIndexMap[i] = allSVsIndex;
        i++;
      }
      svIndexMaps.put(binaryModel, svIndexMap);
    }

// dupe for-block as above?
//    for (BinaryModel<L> binaryModel : oneVsOneModels.values()) {
//      int[] svIndexMap = new int[binaryModel.SVs.length];
//      int i = 0;
//      for (SparseVector p : binaryModel.SVs) {
//        Integer allSVsIndex = allSVsMap.get(p);
//        if (allSVsIndex == null) {
//          allSVsIndex = totalSVs;
//          allSVsMap.put(p, allSVsIndex);
//          totalSVs++;
//        }
//        svIndexMap[i] = allSVsIndex;
//        i++;
//      }
//      svIndexMaps.put(binaryModel, svIndexMap);
//    }

    allSVs = new SparseVector[totalSVs];

    for (Map.Entry<SparseVector, Integer> entry : allSVsMap.entrySet()) {
      allSVs[entry.getValue()] = entry.getKey();
    }
  }

  public synchronized void putOneVsAllModel(L label1, BinaryModel<L> binaryModel) {
    oneVsAllModels.put(label1, binaryModel);
  }

  public synchronized void putOneVsOneModel(L label1, L label2, BinaryModel<L> binaryModel) {
    oneVsOneModels.put(label1, label2, binaryModel);
  }

  protected void readSupportVectors(BufferedReader fp) {
    //BAD Implement support vector I/O
    throw new UnsupportedOperationException();
  }

  protected void writeSupportVectors(DataOutputStream fp) throws IOException {
    fp.writeBytes("SV\n");
    fp.writeBytes("Saving multi-class support vectors is not implemented yet");

    //BAD Implement support vector I/O
    // the original format is a spaghetti
  }

  public void writeToStream(DataOutputStream fp) throws IOException {
    throw new RuntimeException("Not implemented");

  }

  public String getInfo() {
    if (crossValidationResults != null) {
      return crossValidationResults.getInfo();
    } else {
      StringBuffer result = new StringBuffer();
      if (oneVsAllMode != OneVsAllMode.None) {
        Multiset<Double> cs = HashMultiset.create();
        Multiset<KernelFunction> kernels = HashMultiset.create();
        for (BinaryModel<L> binaryModel : oneVsAllModels.values()) {
          cs.add(binaryModel.param.C);
          kernels.add(binaryModel.param.kernel);
        }
        result.append("OneVsAll:C=" + cs + "; gamma=" + kernels + "   ");
      }
      if (allVsAllMode != AllVsAllMode.None) {
        Multiset<Double> cs = HashMultiset.create();
        Multiset<KernelFunction> kernels = HashMultiset.create();
        for (BinaryModel<L> binaryModel : oneVsOneModels.values()) {
          cs.add(binaryModel.param.C);
          kernels.add(binaryModel.param.kernel);
        }
        result.append("AllVsAll:C=" + cs + "; gamma=" + kernels + "   ");
      }
      return result.toString();
    }

  }

// -------------------------- ENUMERATIONS --------------------------

  public enum OneVsAllMode {
    // note all of these modes except None require probability=true
    None, Best, Veto, BreakTies, VetoAndBreakTies
  }

  public enum AllVsAllMode {
    None, AllVsAll, FilteredVsAll, FilteredVsFiltered
  }

// -------------------------- INNER CLASSES --------------------------

  private class SymmetricHashMap2d<K extends Comparable, V> {
// ------------------------------ FIELDS ------------------------------

    HashMap<K, Map<K, V>> l1Map;
    private int sizePerDimension;

    public boolean isEmpty() {
      return l1Map.isEmpty();
    }

    // --------------------------- CONSTRUCTORS ---------------------------

    public SymmetricHashMap2d(SymmetricHashMap2d<K, V> copyFrom, Collection<K> excludeKeys) {
      this(copyFrom.sizePerDimension);// overkill since we're removing some

      for (Map.Entry<K, Map<K, V>> entry : copyFrom.l1Map.entrySet()) {
        K k1 = entry.getKey();
        if (!excludeKeys.contains(k1)) {
          Map<K, V> l2Map = new HashMap<K, V>(
              sizePerDimension); // overkill since we're removing some
          for (Map.Entry<K, V> entry2 : entry.getValue().entrySet()) {
            K k2 = entry2.getKey();
            if (!excludeKeys.contains(k2)) {
              l2Map.put(k2, entry2.getValue());
            }
          }

          l1Map.put(k1, l2Map);
        }
      }
    }

    public SymmetricHashMap2d(int sizePerDimension) {
      this.sizePerDimension = sizePerDimension;
      l1Map = new HashMap<>(sizePerDimension);
    }

// -------------------------- OTHER METHODS --------------------------

    V get(K k1, K k2) {
      if (k1.compareTo(k2) > 0) {
        K k3 = k1;
        k1 = k2;
        k2 = k3;
      }

      Map<K, V> l2Map = l1Map.get(k1);
      if (l2Map == null) {
        l2Map = new HashMap<>(sizePerDimension);
        l1Map.put(k1, l2Map);
      }

      return l2Map.get(k2);
    }

    public Set<K> keySet() {
      // not all keys are represented in the l1Map; in particular, the last one is missing since there are no greater elements.
      Set<K> result = new HashSet<>();
      result.addAll(l1Map.keySet());
      if (!l1Map.isEmpty()) {
        result.addAll(l1Map.values().iterator().next().keySet());
      }
      return result;
    }

    public void put(K k1, K k2, V value) {
      if (k1.compareTo(k2) > 0) {
        K k3 = k1;
        k1 = k2;
        k2 = k3;
      }

      Map<K, V> l2Map = l1Map.get(k1);
      if (l2Map == null) {
        l2Map = new HashMap<>();
        l1Map.put(k1, l2Map);
      }

      l2Map.put(k2, value);
    }

    public Iterable<V> values() {
      return new Iterable<V>() {
        public Iterator<V> iterator() {
          return valueIterator();
        }
      };
    }

    public Iterator<V> valueIterator() {
      return new Iterator<V>() {
        Iterator<K> k1iter = l1Map.keySet().iterator();
        Iterator<V> l2iter = null;

        public boolean hasNext() {
          return (l2iter != null && l2iter.hasNext()) || k1iter.hasNext();
        }

        public V next() {
          if (l2iter == null || !l2iter.hasNext()) {
            if (k1iter.hasNext()) {
              l2iter = l1Map.get(k1iter.next()).values().iterator();
            } else {
              return null;
            }
          }

          return l2iter.next();
        }

        public void remove() {
          throw new UnsupportedOperationException();
        }
      };
    }
  }

  public Collection<L> getLabels() {
    if (oneVsOneModels != null && !oneVsOneModels.isEmpty()) {
      return oneVsOneModels.keySet();
    } else if (oneVsAllModels != null && !oneVsAllModels.isEmpty()) {
      return oneVsAllModels.keySet();
    }
    throw new SvmException(
        "Can't get labels from a MultiClassModel with no subsidiary BinaryModels");
  }
}
