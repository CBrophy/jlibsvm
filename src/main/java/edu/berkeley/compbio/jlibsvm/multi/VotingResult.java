package edu.berkeley.compbio.jlibsvm.multi;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public class VotingResult<L> {
// ------------------------------ FIELDS ------------------------------

  private L bestLabel = null;
  private double bestVoteProportion = 0;
  private double secondBestVoteProportion = 0;
  private double bestOneClassProbability = 0;
  private double secondBestOneClassProbability = 0;
  private double bestOneVsAllProbability = 0;
  private double secondBestOneVsAllProbability = 0;

// --------------------------- CONSTRUCTORS ---------------------------

  public VotingResult() {
  }

  public VotingResult(L bestLabel, double bestVoteProportion, double secondBestVoteProportion,
      double bestOneClassProbability, double secondBestOneClassProbability,
      double bestOneVsAllProbability, double secondBestOneVsAllProbability) {
    this.bestLabel = bestLabel;
    this.bestVoteProportion = bestVoteProportion;
    this.secondBestVoteProportion = secondBestVoteProportion;
    this.bestOneClassProbability = bestOneClassProbability;
    this.secondBestOneClassProbability = secondBestOneClassProbability;
    this.bestOneVsAllProbability = bestOneVsAllProbability;
    this.secondBestOneVsAllProbability = secondBestOneVsAllProbability;
  }

// --------------------- GETTER / SETTER METHODS ---------------------


  public L getBestLabel() {
    return bestLabel;
  }

  public double getBestOneVsAllProbability() {
    return bestOneVsAllProbability;
  }

  public double getBestOneClassProbability() {
    return bestOneClassProbability;
  }

  public double getBestVoteProportion() {
    return bestVoteProportion;
  }

  public double getSecondBestOneVsAllProbability() {
    return secondBestOneVsAllProbability;
  }

  public double getSecondBestOneClassProbability() {
    return secondBestOneClassProbability;
  }

  public double getSecondBestVoteProportion() {
    return secondBestVoteProportion;
  }
}
