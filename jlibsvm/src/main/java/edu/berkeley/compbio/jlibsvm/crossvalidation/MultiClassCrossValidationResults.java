package edu.berkeley.compbio.jlibsvm.crossvalidation;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Imported from https://github.com/davidsoergel/ml
 *
 * LICENSE
 * https://github.com/davidsoergel/ml/blob/master/LICENSE
 */
public class MultiClassCrossValidationResults<L extends Comparable> extends CrossValidationResults
{
  private static final String[] EMPTY_STRING_ARRAY = new String[0];
  protected int numExamples;
  private final Map<L, Multiset<L>> confusionMatrix;

  // BAD test this class

  // BAD currently we don't include these in the computations...
  // samples that had no actual label, and so should be predicted "unknown"
  private final Multiset<L> confusionRowNull = HashMultiset.create();

  //private Map<L, String> friendlyLabelMap;

  public MultiClassCrossValidationResults()//Map<L, String> friendlyLabelMap)
  {

    confusionMatrix = new ConcurrentHashMap<>();

  }


  public SortedSet<L> getLabels()
  {
    return new TreeSet<L>(confusionMatrix.keySet());
  }


  public String[] getFriendlyLabels(final Map<L, String> friendlyLabelMap)
  {
    if (friendlyLabelMap == null)
    {
      return null;
    }
    final List<String> result = new ArrayList<String>(confusionMatrix.size());
    for (final L l : getLabels())
    {
      result.add(friendlyLabelMap.get(l));
    }
    return result.toArray(EMPTY_STRING_ARRAY);
  }

  public void sanityCheck()
  {
    int predictionCount = 0;
    for (final Multiset<L> ls : confusionMatrix.values())
    {
      predictionCount += ls.size();
    }
    assert predictionCount == numExamples;  // every example got a prediction (perhaps null)
  }


  public void addSample(final L realValue, final L predictedValue)
  {
    final Multiset<L> confusionRow = realValue == null ? confusionRowNull : confusionMatrix.computeIfAbsent(realValue, l ->HashMultiset.create());
    confusionRow.add(predictedValue);
    numExamples++;
  }

  public float accuracy()
  {
    int correct = 0;
    for (final Map.Entry<L, Multiset<L>> entry : confusionMatrix.entrySet())
    {
      correct += entry.getValue().count(entry.getKey());
    }
    return (float) correct / (float) numExamples;
  }

  public float unknown()
  {
    int unknown = 0;
    for (final Map.Entry<L, Multiset<L>> entry : confusionMatrix.entrySet())
    {
      unknown += entry.getValue().count(null);
    }
    return (float) unknown / (float) numExamples;
  }

  public float accuracyGivenClassified()
  {
    int correct = 0;
    int unknown = 0;
    for (final Map.Entry<L, Multiset<L>> entry : confusionMatrix.entrySet())
    {
      correct += entry.getValue().count(entry.getKey());
      unknown += entry.getValue().count(null);
    }
    return (float) correct / ((float) numExamples - (float) unknown);
  }

  public float sensitivity(final L label)
  {
    final Multiset<L> predictionsForLabel = confusionMatrix.computeIfAbsent(label, l ->HashMultiset.create());
    final int totalWithRealLabel = predictionsForLabel.size();
    final int truePositives = predictionsForLabel.count(label);
    return (float) truePositives / (float) totalWithRealLabel;
  }

  public float precision(final L label)
  {
    final Multiset<L> predictionsForLabel = confusionMatrix.computeIfAbsent(label, l ->HashMultiset.create());

    final int truePositives = predictionsForLabel.count(label);
    final float total = (float) getTotalPredicted(label);
    return total == 0 ? 1f : (float) truePositives / total;
  }

  public float[] getSpecificities()
  {
    final float[] result = new float[confusionMatrix.size()];
    int i = 0;
    for (final L label : getLabels())
    {
      result[i] = specificity(label);
      i++;
    }
    return result;
  }

  public float[] getSensitivities()
  {
    final float[] result = new float[confusionMatrix.size()];
    int i = 0;
    for (final L label : getLabels())
    {
      result[i] = sensitivity(label);
      i++;
    }
    return result;
  }

  public float[] getPrecisions()
  {
    final float[] result = new float[confusionMatrix.size()];
    int i = 0;
    for (final L label : getLabels())
    {
      result[i] = precision(label);
      i++;
    }
    return result;
  }

  public float[] getPredictedCounts()
  {
    final float[] result = new float[confusionMatrix.size()];
    int i = 0;
    for (final L label : getLabels())
    {
      result[i] = getTotalPredicted(label);
      i++;
    }
    return result;
  }

  public float[] getActualCounts()
  {
    final float[] result = new float[confusionMatrix.size()];
    int i = 0;
    for (final L label : getLabels())
    {
      result[i] = getTotalActual(label);
      i++;
    }
    return result;
  }

  public int getCount(L actual, L predicted)
  {
    return confusionMatrix.computeIfAbsent(actual, l ->HashMultiset.create()).count(predicted);
  }

  public float specificity(final L label)
  {
    // == sensitivity( not label )
    // note "unknown" counts as a negative

    final Multiset<L> predictionsForLabel = confusionMatrix.computeIfAbsent(label, l ->HashMultiset.create());

    final int hasLabel = predictionsForLabel.size();
    final int hasLabelRight = predictionsForLabel.count(label);  // true positives


    final int notLabelWrong = getTotalPredicted(label) - hasLabelRight;  // false negatives
    final int notLabel = numExamples - hasLabel;
    final int notLabelRight = notLabel - notLabelWrong;   // true negatives

    if (notLabel == 0)
    {
      return 1.0f;
    }

    return (float) notLabelRight / (float) notLabel;
  }

  public int getTotalPredicted(final L label)

  {
    int totalWithPredictedLabel = 0;

    // PERF if we want precisions for all the labels, it's inefficient to iterate this n times; in practice it doesn't matter though since there are few enough labels
    for (final Map.Entry<L, Multiset<L>> entry : confusionMatrix.entrySet())
    {
      totalWithPredictedLabel += entry.getValue().count(label);
    }
    return totalWithPredictedLabel;
  }

  public int getTotalActual(final L label)
  {
    if (label == null)
    {
      return confusionRowNull.size();
    }
    else
    {
      return confusionMatrix.computeIfAbsent(label, l ->HashMultiset.create()).size();
    }
  }

  public float classNormalizedSpecificity()
  {
    float sum = 0;
    for (final L label : confusionMatrix.keySet())
    {
      sum += specificity(label);
    }
    return sum / (float) confusionMatrix.size();
  }

  public float classNormalizedSensitivity()
  {
    float sum = 0;
    for (final L label : confusionMatrix.keySet())
    {
      sum += sensitivity(label);
    }
    return sum / (float) confusionMatrix.size();
  }

  public float classNormalizedPrecision()
  {
    float sum = 0;
    for (final L label : confusionMatrix.keySet())
    {
      final float v = precision(label);
      if (!Double.isNaN(v))
      {
        sum += v;
      }
      else
      {
        Logger.getGlobal().warning("Label " + label + " did not contribute to precision; " + getTotalPredicted(label)
            + " predictions");
      }
    }
    return sum / (float) confusionMatrix.size();
  }

  public int numPopulatedRealLabels()
  {
    return confusionMatrix.size();
  }

  public int numPredictedLabels()
  {
    final Set<L> predictedLabels = new HashSet<L>();
    for (final Multiset<L> ls : confusionMatrix.values())
    {
      predictedLabels.addAll(ls.elementSet());
    }
    return predictedLabels.size();
  }
}

