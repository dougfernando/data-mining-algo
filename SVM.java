import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Implementation of SVM - Support vector machine (Batch, SGD, MiniBatch)
 * @author Douglas Fernando da Silva - doug.fernando@gmail.com
 */
public class SVM {
	// Parameter structure for Batch, SGD and MiniBatch
	static class SVMParameter {
		public int batchSize;
		public double eta;
		public double eps;
		public int c;

		public SVMParameter(int batchSize, double eta, double eps, int c) {
			this.batchSize = batchSize;
			this.eta = eta;
			this.eps = eps;
			this.c = c;
		}
	}
	
	// Structure to hold the result of the SVM
	static class SVMResult {
		public double[] w;
		public double b;
		public SVMResult(double[] w, double b) {
			this.w = w;
			this.b = b;
		}
	}
	
	public static void main(String[] args) throws Exception {
		SVM svm = new SVM();
 
		double[][] xi = svm.readAsVectors(args[0]); // features
		double[] yi = svm.readAsVectors(args[1])[0]; // target

//		svm.runBatch(xi, yi);
//		svm.runSGD(xi, yi);
		svm.runMiniBatch(xi, yi);
	    
		double[][] datasetTrain = svm.readAsVectors(args[2]);
		double[] fxTrain = svm.readAsVectors(args[3])[0];
		double[][] datasetTest = svm.readAsVectors(args[4]);
		double[] fxTest = svm.readAsVectors(args[5])[0];

//		svm.runRegParamAnalysis(datasetTrain, fxTrain, datasetTest, fxTest);
	    
		System.exit(0);
	}

	public void runRegParamAnalysis(double[][] datasetTrain, double[] fxTrain, double[][] datasetTest,
			double[] fxTest) {
		// REG PARAM
	    int[] c = {1, 10, 50, 100, 200, 300, 400, 500 };
	    for (int i = 0; i < c.length; i++) {
		    SVMResult r = miniBatchGradDesc(datasetTrain, fxTrain, new SVMParameter(1, 0.0001, 0.001, c[i]));
			double percError = calcPercError(r, datasetTest, fxTest);
			System.out.println(String.format("C: %d | Perc. Error: %.2f", c[i], percError));
		}
	}

	// MINI BATCH
	public void runMiniBatch(double[][] dataset, double[] fx) {
		long startTime = System.currentTimeMillis();
		miniBatchGradDesc(dataset, fx, new SVMParameter(20, 0.00001, 0.01, 100));
		long stopTime = System.currentTimeMillis();
	    System.out.println("Mini Batch elapsed time: " + (stopTime - startTime));
	}

	// SGD
	public void runSGD(double[][] dataset, double[] fx) {
		long startTime = System.currentTimeMillis();
		miniBatchGradDesc(dataset, fx, new SVMParameter(1, 0.0001, 0.001, 100)); // SGD => batch size = 1
		long stopTime = System.currentTimeMillis();
	    System.out.println("SGD elapsed time: " + (stopTime - startTime));
	}

	// BATCH
	public void runBatch(double[][] dataset, double[] fx) {
		long startTime = System.currentTimeMillis();
		batchGradient(dataset, fx, new SVMParameter(1, 0.0000003, 0.25, 100));
		long stopTime = System.currentTimeMillis();
	    System.out.println("Batch gradient elapsed time: " + (stopTime - startTime));
	}
	
	// Percentual error for item f)
	private double calcPercError(SVMResult r, double[][] datasetTest, double[] fxTest) {
		int n = datasetTest.length;
		int errorCount = 0;
		for (int i = 0; i < fxTest.length; i++) {
			double dotValue = dot(datasetTest[i], r.w);
			double val = fxTest[i] * (dotValue  + r.b);
			errorCount += val < 0? 1: 0;
		}
		
		double result = (double)errorCount/n;
		return result;
	}

	// For batchSize = 1 => SGD
	public SVMResult miniBatchGradDesc(double[][] xi, double[] yi, SVMParameter svmParam) {
		int d = xi[0].length; // number of dimensions
		double[] w = new double[d]; // w for the SVM (result)
		double b = 0; // b for the SVM (result)
		boolean shouldStop = false;
		int n = xi.length; // number of training samples
		double deltaCostMinus1 = 0; // delta cost for k-1 iteration
		shuffle(xi, yi); // shuffle the training date
		double fkMinus1 = calcFk(w, b, xi, yi, svmParam.c); // f0
		
		int k = 0;
		int l = 0;
		while (!shouldStop) {			
			int ini = l * svmParam.batchSize;
			int end = Math.min(n, (l + 1) * svmParam.batchSize);
			
			double[] wNew = new double[d]; // for updating w using w from previous iteration
			for (int j = 0; j < d; j++) {
				double tmp = w[j] - svmParam.eta * gradW(j, w, b, xi, yi, ini, end, svmParam.c);
				wNew[j] = tmp;
			}
			w = wNew;
			
			b += (-1.0 * svmParam.eta * gradB(w, b, xi, yi, ini, end, svmParam.c));
			l = (l + 1) % ((n + svmParam.batchSize - 1) / svmParam.batchSize);
			
			double fk = calcFk(w, b, xi, yi, svmParam.c);
			double deltaCostPerc = deltaCostPerc(fk, fkMinus1);
			double newDeltaCost = 0.5 * deltaCostMinus1 + 0.5 * deltaCostPerc;
			
			shouldStop = newDeltaCost < svmParam.eps;
			deltaCostMinus1 = newDeltaCost;
			fkMinus1 = fk;
			System.out.println(String.format("SGD - K: %d | FK: %.4f | Error: %.4f | ini: %s | end: %s | b: %.4f | |w|: %.4f", k, fk, newDeltaCost, ini, end, b, dot(w,w)));
			k++;
		}

		return new SVMResult(w, b);
	}

	private void shuffle(double[][] xi, double[] yi) {
		Random rgen = new Random(System.currentTimeMillis());
		for (int i = 0; i < xi.length; i++) {
		    int randomPosition = rgen.nextInt(xi.length);
		    double[] temp = xi[i];
		    xi[i] = xi[randomPosition];
		    xi[randomPosition] = temp;
		    double fxTmp = yi[i];
		    yi[i] = yi[randomPosition];
		    yi[randomPosition] = fxTmp;
		}
	}
	
	public SVMResult batchGradient(double[][] xi, double[] yi, SVMParameter svmParam) {
		int d = xi[0].length;
		double[] w = new double[d];
		double b = 0;
		int k = 0;
		boolean shouldStop = false;
		int n = xi.length;
		double fkMinus1 = calcFk(w, b, xi, yi, svmParam.c); // f0

		while (!shouldStop) {
			double[] wNew = new double[d];
			for (int j = 0; j < d; j++) { // update using w from prev iteration
				double tmp = w[j] - svmParam.eta * gradW(j, w, b, xi, yi, 0, n, svmParam.c);
				wNew[j] = tmp;
			}
			w = wNew; // update w
			
			b += (-1.0 * svmParam.eta * gradB(w, b, xi, yi, 0, n, svmParam.c));

			double fk = calcFk(w, b, xi, yi, svmParam.c);
			double deltaCostPerc = deltaCostPerc(fk, fkMinus1);
			shouldStop = deltaCostPerc < svmParam.eps;
			System.out.println(String.format("Batch - K: %d | FK: %.2f | Error: %.2f ", k, fk, deltaCostPerc));
			fkMinus1 = fk;
			k++;
		}

		return new SVMResult(w, b);
	}

	private double deltaCostPerc(double fk, double fkMinus1) {
		if (fkMinus1 == Double.NEGATIVE_INFINITY)
			return 1.0;

		return Math.abs(fkMinus1 - fk) * 100 / fkMinus1;
	}

	private double calcFk(double[] w, double b, double[][] xi, double[] yi, int C) {
		int n = xi.length;
		double sumWj2 = dot(w, w);

		double sumL = 0;
		for (int i = 0; i < n; i++) {
			double wx = dot(w, xi[i]);
			double h = 1 - yi[i] * (wx + b);
			sumL += C * Math.max(0, h);
		}

		double result = 0.5 * sumWj2 + sumL;
		return result;
	}

	private double gradB(double[] w, double b, double[][] xi, double[] yi, int ini, int end, int C) {
		double sumN = 0.0;
		for (int i = ini; i < end; i++) {
			double classResult = yi[i] * (dot(xi[i], w) + b);
			if (classResult < 1) {
				sumN += -yi[i];
			}
		}

		double result = C * sumN;
		return result;
	}

	private double gradW(int j, double[] w, double b, double[][] xi, double[] yi, int ini, int end, int C) {
		double sumN = 0.0;
		for (int i = ini; i < end; i++) {
			double classResult = yi[i] * (dot(xi[i], w) + b);
			if (classResult < 1) {
				sumN += -yi[i] * xi[i][j];
			}
		}

		double result = w[j] + C * sumN;
		return result;
	}

	// dot product between two vectors
	private double dot(double[] a, double[] b) {
		double sum = 0;
		for(int i = 0; i < a.length; i++){
			sum += a[i] * b[i];
		}
		return sum;
	}

	// read the input files (features and target)
	public double[][] readAsVectors(String filePath) throws IOException {
		ArrayList<double[]> result = new ArrayList<double[]>();
		FileReader fileReader = new FileReader(filePath);
		BufferedReader br = new BufferedReader(fileReader);
		String line;
		while ((line = br.readLine()) != null) {
			String[] data = line.split(",");
			double[] lineData = new double[data.length];
			for (int i = 0; i < data.length; i++) {
				lineData[i] = Double.valueOf(data[i]);
			}

			result.add(lineData);
		} br.close();

		double[][] finalResult = null;
		if (result.get(0).length == 1) { // TARGET
			int rows = result.size();

			finalResult = new double[1][rows]; // transpose
			for (int i = 0; i < rows; i++) {
				finalResult[0][i] = result.get(i)[0];
			}

			return finalResult;
		}

		// FEATURES
		int rows = result.size();
		int cols = result.get(0).length;
		finalResult = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			finalResult[i] = result.get(i);
		}

		return finalResult;
	}
}
