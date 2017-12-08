package edu.berkeley.compbio.jlibsvm.binary;

import com.davidsoergel.dsutils.DSArrayUtils;
import edu.berkeley.compbio.jlibsvm.SolutionModel;
import edu.berkeley.compbio.jlibsvm.util.SparseVector;
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class AlphaModel<L extends Comparable> extends SolutionModel<L> {
// ------------------------------ FIELDS ------------------------------

  // used only during training, then ditched
  public Map<SparseVector, Double> supportVectors;

  // more compact representation used after training
  public int numSVs;
  public SparseVector[] SVs;
  public double[] alphas;

  public double rho;

// --------------------------- CONSTRUCTORS ---------------------------

  protected AlphaModel() {
    super();
  }



// -------------------------- OTHER METHODS --------------------------

  /**
   * Remove vectors whose alpha is zero, leaving only support vectors
   */
  public void compact() {
    // do this first so as to make the arrays the right size below
    supportVectors.entrySet().removeIf(entry -> entry.getValue() == 0);

    // put the keys and values in parallel arrays, to free memory and maybe make things a bit faster (?)

    numSVs = supportVectors.size();
    SVs = new SparseVector[numSVs];
    alphas = new double[numSVs];

    int c = 0;
    for (Map.Entry<SparseVector, Double> entry : supportVectors.entrySet()) {
      SVs[c] = entry.getKey();
      alphas[c] = entry.getValue();
      c++;
    }

    supportVectors = null;
  }

  protected void readSupportVectors(BufferedReader reader) throws IOException {

    List<Double> alphaList = new ArrayList<Double>();
    List<SparseVector> svList = new ArrayList<SparseVector>();

    String line;
    while ((line = reader.readLine()) != null) {

      final String trimmed = line.trim();

      int question = trimmed.indexOf('?');

      if (question < 0) {
        continue;
      }

      alphaList.add(Double.parseDouble(trimmed.substring(0, question)));

      SparseVector vector = SparseVector.fromString(trimmed.substring(question + 1));

      if (vector != null) {
        svList.add(vector);
      }
    }

    alphas = DSArrayUtils.toPrimitiveDoubleArray(alphaList);
    SVs = svList.toArray(new SparseVector[0]);

    numSVs = SVs.length;

    supportVectors = null; // we read it directly to the compact representation
  }

  protected void writeSupportVectors(DataOutputStream fp) throws IOException {
    fp.writeBytes("SV\n");

    for (int i = 0; i < numSVs; i++) {

      fp.writeBytes(alphas[i] + "?");

      fp.writeBytes(SVs[i].toString());

      fp.writeBytes("\n");
    }
  }

  public void writeToStream(DataOutputStream fp) throws IOException {
    super.writeToStream(fp);

    fp.writeBytes("rho " + rho + "\n");
    fp.writeBytes("total_sv " + numSVs + "\n");
  }
}
