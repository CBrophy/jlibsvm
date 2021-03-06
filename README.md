jLibSVM
=======

_Efficient training of Support Vector Machines in Java_

 * Heavily refactored Java port of the venerable [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) (version 2.88).
 * Provides **idiomatic Java** class structure and APIs (unlike the Java version provided by LIBSVM, which is transliterated C code).
 * Easy to **add new kernels**, in addition to the five standard ones provided by LIBSVM.
 * On the mathematical side, jlibsvm performs **exactly the same computations as LIBSVM**, including shrinking and all the fancy stuff described in the [LIBSVM implementation docs](http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).
 * **Optimized kernel implementations** run faster, particularly when input vectors are sparse.  For instance, on the [mushrooms](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms) dataset, jlibsvm trained ~25% faster than LIBSVM (java version) with an RBF kernel and ~40% faster with a linear kernel.  (The C version of LIBSVM is still faster, though).
 * **Multithreaded training** to take advantage of modern multi-core machines (using Java8 Streams).
 * **Integrated scaling and normalization** so you don't have to explicitly preprocess your data.
 * **Integrated grid search** for optimal kernel parameters.
 * Drop-in replacement if you use the command-line tools (e.g. svm-train, etc.), but not if you use LIBSVM programmatically.
 * Uses Java **generics** throughout, including for classification labels, so you can specify that the "label" of a class be of whatever Java type you like.  In an email-filtering application, for example, you could use objects of type `Mailbox` as the labels.  That would allow you to write something like `mySvmSolutionModel.predict(incomingEmail).addMessage(incomingEmail)`.  The `predict()` method returns a classification label, which in this case is an object of class `Mailbox`, which has an `addMessage()` method.
 

Status
------

[cbrophy]

Lots of updates for a production environment.
* Removed commented-out code, added some tests and removed extra templating
* Updates to SparseVector to make computations easier to read and test
* float -> double
* Updated to java 8 spec
* Removed dependencies that were replicating java 8 functionality
* Updated dependencies (moved from google collections to guava)
* Moved from maven to gradle
* Other bug fixes, changes, and tweaks (check the comment history)
* CAUTION: This fork has several breaking changes from the original and may not be an easy port for existing implementations

[davidsoergel]

This is beta code.  While LIBSVM is stable, it's possible that I broke something in the process of refactoring it.  I've done ad-hoc testing primarily with the C_SVC machine and an RBF kernel, and got results that were identical to LIBSVM as far as I could tell.  There are not (yet?) any unit tests.  I'm running some automated verifications that jlibsvm behaves identically to LIBSVM for a number of input datasets and parameter choices; results will be available here soon.  Please file a bug if you find a situation in which the two packages give different results.

Documentation
-------------


Sorry, I haven't really had a chance to write any docs.  Have a look at the sources for the command-line programs in the [legacyexec](src/main/java/edu/berkeley/compbio/jlibsvm/legacyexec) package to see how jlibsvm gets called.  Very briefly, you'll need to:

 1. instantiate the [KernelFunction](http://davidsoergel.github.io/jlibsvm/apidocs/edu/berkeley/compbio/jlibsvm/kernel/KernelFunction.html) that you want
 2. set up some parameters in a new [SvmParameter](http://davidsoergel.github.io/jlibsvm/apidocs/edu/berkeley/compbio/jlibsvm/SvmParameter.html) object
 3. instantiate a concrete subclass of [SvmProblem](http://davidsoergel.github.io/jlibsvm/apidocs/edu/berkeley/compbio/jlibsvm/SvmProblem.html) (binary, multiclass, or regression), and populate it with training data
 4. instantiate a concrete subclass of [SVM](http://davidsoergel.github.io/jlibsvm/apidocs/edu/berkeley/compbio/jlibsvm/SVM.html), choosing a type appropriate for your problem
 5. Call `SVM.train(problem)` to yield a [SolutionModel](http://davidsoergel.github.io/jlibsvm/apidocs/edu/berkeley/compbio/jlibsvm/SolutionModel.html), which can be used to make predictions
