package edu.berkeley.compbio.jlibsvm;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.io.Files;
import edu.berkeley.compbio.jlibsvm.crossvalidation.CrossValidationResults;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Collection;


/**
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */
public abstract class SolutionModel<L extends Comparable> implements Serializable {
// ------------------------------ FIELDS ------------------------------

  public abstract CrossValidationResults getCrossValidationResults();

// -------------------------- STATIC METHODS --------------------------

  public static <L extends Comparable, T extends SolutionModel<L>> T load(String modelFileName,
      Class<T> modelType) {
    checkNotNull(modelFileName, "modelFileName");
    checkNotNull(modelType, "modelType");

    File file = new File(modelFileName);

    checkArgument(file.exists(), "Model file not found: %s", file.getAbsolutePath());

    T result = null;
    ObjectMapper objectMapper = new ObjectMapper();

    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(new FileInputStream(file)))) {
      String line = null;
      while ((line = reader.readLine()) != null) {
        checkState(result == null, "Model file expected to contain only one model");

        result = objectMapper.readValue(line, modelType);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return result;
  }

// --------------------------- CONSTRUCTORS ---------------------------

  public SolutionModel() {
  }

// -------------------------- OTHER METHODS --------------------------

  public void save(String modelFileName) throws IOException {
    checkNotNull(modelFileName, "modelFileName");
    ObjectMapper objectMapper = new ObjectMapper();

    File file = new File(modelFileName);

    Files.createParentDirs(file);

    if (file.exists()) {
      checkState(file.delete(), "Failed to delete existing file: %s", file.getAbsolutePath());
    }

    try (BufferedWriter writer = new BufferedWriter(
        new OutputStreamWriter(new FileOutputStream(file)))) {
      writer.write(objectMapper.writeValueAsString(this));
    }

  }


  public String getKernelName() {
    throw new RuntimeException("Not implemented");
  }

  public Collection<L> getLabels() {
    throw new RuntimeException("Not implemented");
  }
}
