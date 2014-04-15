import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * K-means clustering implementation using Hadoop
 * 
 * @author Douglas Fernando da Silva - doug.fernando@gmail.com
 * 
 */
public class KMeans extends Configured implements Tool {
	private static final int MAX_ITERATIONS = 20;

	public interface Consts {
		public static final String INPUT_PATH = "data";
		public static final String OUTPUT_PATH = "outputkmeans";
		public static final String CENTROID_KEY = "centroid.dfs";
		public static final String CENTROID1_PATH = "centroid1.txt";
		public static final String CENTROID2_PATH = "centroid2.txt";
		public static final String ITERATION_KEY = "iteration.dfs";
		public static final String COST_PATH = "cost.txt";
	}

	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		int res = ToolRunner.run(new Configuration(), new KMeans(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		int currentIteration = 0;

		// Delete the cost output file
		Path costPath = new Path(Consts.COST_PATH);
		FileSystem fs = FileSystem.get(new Configuration());
		if (fs.exists(costPath)) {
			fs.delete(costPath, true);
		}
		
		// Run the algorithm for MAX_ITERATIONS iterations
		while (currentIteration < MAX_ITERATIONS) {
			System.out.println("Iteration: " + currentIteration);
			Job job = createJob(args, currentIteration);
			job.waitForCompletion(true);
			currentIteration++;
		}

		return 0;
	}

	private Job createJob(String[] args, int iteration) throws IOException {
		Configuration conf = new Configuration();

		// Decides which initial centroid file to use
		String centroidPath = "2".equals(args[0]) ? Consts.CENTROID2_PATH : Consts.CENTROID1_PATH;
		
		conf.set(Consts.CENTROID_KEY, centroidPath);

		String iterationAsString = String.valueOf(iteration);
		conf.set(Consts.ITERATION_KEY, iterationAsString);

		Job job = new Job(conf, "KMeans-" + iteration);
		job.setJarByClass(KMeans.class);

		String inputURL = Consts.INPUT_PATH;
		String outputURL = Consts.OUTPUT_PATH + iterationAsString;

		Path input = new Path(inputURL + ".txt");
		Path output = new Path(outputURL + ".txt");

		// Delete output from previous executions
		FileSystem fs = FileSystem.get(conf);
		if (fs.exists(output)) {
			fs.delete(output, true);
		}

		FileInputFormat.addInputPath(job, input);
		FileOutputFormat.setOutputPath(job, output);

		// Job types set up
		job.setMapOutputKeyClass(VectorWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);

		job.setOutputKeyClass(VectorWritable.class);
		job.setOutputValueClass(VectorWritable.class);

		job.setMapperClass(MapJob.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		return job;
	}

	/**
	 * Vector representation (centroid and items) - hadoop compatible
	 */
	static class VectorWritable implements WritableComparable<VectorWritable> {
		private ArrayList<Float> vector;

		private VectorWritable(List<Float> vector) {
			this.vector = new ArrayList<Float>(vector);
		}

		public VectorWritable(Float[] vector) {
			this(Arrays.asList(vector));
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(vector.size());
			for (Float item : vector) {
				out.writeFloat(item);
			}
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			Integer size = in.readInt();
			vector = new ArrayList<Float>();
			for (int i = 0; i < size; i++) {
				vector.add(in.readFloat());
			}
		}

		public int size() {
			return vector.size();
		}

		public VectorWritable() {
		}

		public Float get(int index) {
			return vector.get(index);
		}

		public Float[] asFloatArray() {
			return vector.toArray(new Float[0]);
		}

		@Override
		public String toString() {
			DecimalFormat df = new DecimalFormat("#.##");
			String result = "";
			for (Float v : vector) {
				result += df.format(v) + " ";
			}

			return result;
		}

		@Override
		public int hashCode() {
			return toString().hashCode();
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			VectorWritable other = (VectorWritable) obj;
			return toString().equals(other.toString());
		}

		@Override
		public int compareTo(VectorWritable o) {
			VectorWritable o1 = (VectorWritable) o;
			if (toString().equals(o1.toString()))
				return 0;

			double n1 = 0;
			double n2 = 0;

			for (int i = 0; i < vector.size(); i++) {
				n1 += vector.get(i).floatValue() * vector.get(i).floatValue();
				n2 += o1.get(i).floatValue() * o1.get(i).floatValue();
			}

			n1 = Math.sqrt(n1);
			n2 = Math.sqrt(n2);

			return n1 > n2 ? 1 : -1;
		}
	}

	public static class MapJob extends Mapper<LongWritable, Text, VectorWritable, VectorWritable> {
		private List<Float[]> centroids = new ArrayList<Float[]>();
		private float totalCost = 0;

		/**
		 * Reads the centroid file and make the information available for using the map method
		 */
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path centroidsPath = new Path(conf.get(Consts.CENTROID_KEY));
			FileSystem fs = FileSystem.get(conf);
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(centroidsPath)));
			String line = br.readLine();
			System.out.println("Reading centroids...");
			while (line != null) {
				Float[] data = toFloatArray(line);
				centroids.add(data);
				line = br.readLine();
			}
			System.out.println("Number of centroids read:" + centroids.size());
			br.close();
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			String iteration = context.getConfiguration().get(Consts.ITERATION_KEY);
			String costPhiStr = String.valueOf(totalCost);
			writeCostToFile(context, iteration, costPhiStr);
		}

		private void writeCostToFile(Context context, String iteration, String costPhi) throws IOException {
			// Read the current content of the file to rewrite later
			List<String> currentContent = new ArrayList<String>();
			Configuration conf = context.getConfiguration();
			Path costPath = new Path(Consts.COST_PATH);
			FileSystem fs = FileSystem.get(conf);
			if (fs.exists(costPath)) {
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(costPath)));
				String line;
				while ((line = br.readLine()) != null) {
					currentContent.add(line);
				}
				br.close();
				
				fs.delete(costPath, true);
			}
			
			// rewrite existing content and add new cost
			BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(costPath, true)));
			for (String line : currentContent) {
				br.write(line + "\n");
			}
			br.write("Iteration: " + iteration + " Cost: " + costPhi + "\n");
			br.close();
		}

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			Float[] result = null;
			Float distance = Float.MAX_VALUE;

			Float[] inputAsFloat = toFloatArray(value.toString());

			for (Float[] f : centroids) { // find which is the closest centroid
				Float tmpDist = l2distance(f, inputAsFloat);

				if (tmpDist < distance) {
					result = f;
					distance = tmpDist;
				}
			}

			totalCost += distance;

			VectorWritable nearestCentroid = new VectorWritable(result);

			VectorWritable currentItem = new VectorWritable(inputAsFloat);

			// Emits: Nearest centroid -> Item
			context.write(nearestCentroid, currentItem);
		}

		/**
		 * Read as input a line with of space separated floats values and return an array representing them
		 */
		private Float[] toFloatArray(String value) {
			String[] input = value.split("\\s+");
			Float[] inputAsFloat = new Float[input.length];
			for (int i = 0; i < input.length; i++) {
				inputAsFloat[i] = Float.parseFloat(input[i]);
			}
			return inputAsFloat;
		}

		/**
		 * Calculates the distance between a centroid and a given item
		 */
		static private Float l2distance(Float[] centroid, Float[] item) {
			double sum = 0.0f;

			for (int i = 0; i < centroid.length; i++) {
				sum += Math.pow((centroid[i] - item[i]), 2.0);
			}

			// cost defined is squared
			return (float) sum;
		}
	}

	public static class Reduce extends Reducer<VectorWritable, VectorWritable, VectorWritable, VectorWritable> {
		List<VectorWritable> centroids = new ArrayList<VectorWritable>();

		/**
		 * Persists the centroids for the next iteration
		 */
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			Configuration conf = context.getConfiguration();
			Path centroidsPath = new Path(conf.get(Consts.CENTROID_KEY));
			Path centroidsPathOld = new Path(conf.get(Consts.CENTROID_KEY) + ".old");

			// delete centroid file from previous iteration
			FileSystem fs = FileSystem.get(conf);
			if (fs.exists(centroidsPathOld))
				fs.delete(centroidsPathOld, true);
			fs.rename(centroidsPath, centroidsPathOld);

			// writes a new file using the centroids previously found using by
			// the reduce
			BufferedWriter br = new BufferedWriter(new OutputStreamWriter(fs.create(centroidsPath, true)));
			System.out.println("Writing centroids...");
			for (VectorWritable c : centroids) {
				br.write(c.toString() + "\n");
			}

			System.out.println("Number of centroids written:" + centroids.size());

			br.close();
		}

		@Override
		public void reduce(VectorWritable key, Iterable<VectorWritable> values, Context context) throws IOException,
				InterruptedException {
			// Find the new centroid for the values associated to the current centroid (key)
			// New centroid is defined by the average of each dimension for all values
			Float[] newCentroidData = new Float[key.size()];
			int itemsCounter = 0;
			for (VectorWritable item : values) {
				// emit the cluster data formed in the map phase
				context.write(key, item);

				for (int i = 0; i < item.size(); i++) {
					newCentroidData[i] = newCentroidData[i] == null ? item.get(i) : newCentroidData[i] + item.get(i);
				}
				itemsCounter++;
			}
			for (int i = 0; i < newCentroidData.length; i++) {
				newCentroidData[i] = newCentroidData[i] / (float) itemsCounter;
			}

			VectorWritable newCentroid = new VectorWritable(newCentroidData);
			centroids.add(newCentroid);
		}
	}
}
