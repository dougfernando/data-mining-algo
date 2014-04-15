import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
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
 * Find potential friends in social networks using Hadoop
 * 
 * @author Douglas Fernando da Silva - doug.fernando@gmail.com
 *
 */
public class FindPotentialFriends extends Configured implements Tool {
	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		int res = ToolRunner.run(new Configuration(), new FindPotentialFriends(), args);

		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));

		Job job = new Job(getConf(), "FindPotentialFriends");
		job.setJarByClass(FindPotentialFriends.class);

		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(CustomEmitWritable.class);

		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);

		job.setMapperClass(MapJob.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.waitForCompletion(true);

		return 0;
	}

	static class CustomEmitWritable implements Writable {
		private Long userId;
		private Boolean isAlreadyConnected;

		public CustomEmitWritable(Long userId, Boolean isAlreadyConnected) {
			this.userId = userId;
			this.isAlreadyConnected = isAlreadyConnected;
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeLong(userId);
			out.writeBoolean(isAlreadyConnected);
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			userId = in.readLong();
			isAlreadyConnected = in.readBoolean();
		}

		public CustomEmitWritable() {
		}
	}

	public static class MapJob extends Mapper<LongWritable, Text, LongWritable, CustomEmitWritable> {

		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] idAndFriends = value.toString().split("\t");

			if (idAndFriends.length > 1) {
				String[] friends = idAndFriends[1].split(",");
				Long[] friendsIds = new Long[friends.length];

				// Signaling existing connections
				for (int i = 0; i < friends.length; i++) {
					friendsIds[i] = Long.parseLong(friends[i]);
					context.write(new LongWritable(Long.parseLong(idAndFriends[0])), new CustomEmitWritable(
							friendsIds[i], true));
				}

				// if users i and j are friends with userId, they may be each
				// others friends
				for (int i = 0; i < friendsIds.length; i++) {
					for (int j = i + 1; j < friendsIds.length; j++) {
						context.write(new LongWritable(Long.parseLong(friends[i])), new CustomEmitWritable(
								friendsIds[j], false));
						context.write(new LongWritable(Long.parseLong(friends[j])), new CustomEmitWritable(
								(friendsIds[i]), false));
					}
				}
			}
		}
	}

	public static class Reduce extends Reducer<LongWritable, CustomEmitWritable, Text, Text> {

		/**
		 * Sort map based on its values
		 */
		public static <K extends Comparable<? super K>, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> map) {
			List<Map.Entry<K, V>> list = new LinkedList<Map.Entry<K, V>>(map.entrySet());
			Collections.sort(list, new Comparator<Map.Entry<K, V>>() {
				public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
					int result = o2.getValue().compareTo(o1.getValue());
					if (result == 0) result = o1.getKey().compareTo(o2.getKey());
					return result;
				}
			});

			Map<K, V> result = new LinkedHashMap<K, V>();
			for (Map.Entry<K, V> entry : list) {
				result.put(entry.getKey(), entry.getValue());
			}
			return result;
		}

		@Override
		public void reduce(LongWritable key, Iterable<CustomEmitWritable> values, Context context) throws IOException,
				InterruptedException {
			HashMap<Long, Integer> recommendationMap = new HashMap<Long, Integer>();

			for (CustomEmitWritable recommendation : values) {
				if (recommendation.isAlreadyConnected) {
					recommendationMap.put(recommendation.userId, 0);
				} else {
					Integer currentCount = recommendationMap.get(recommendation.userId);
					if (currentCount == null) {
						currentCount = 0;
					} else if (currentCount == 0) { // isAlready connected
						continue;
					}

					Integer newValue = currentCount + 1;
					recommendationMap.put(recommendation.userId, newValue);
				}
			}

			Map<Long, Integer> finalValues = sortByValue(recommendationMap);

			Integer maxCount = 0;
			String output = "";
			for (Map.Entry<Long, Integer> entry : finalValues.entrySet()) {
				if (maxCount < 10 && !entry.getValue().equals(0)) {
//					output += String.valueOf(entry.getKey()) + " [" + entry.getValue() + "] " + ",";
					output += String.valueOf(entry.getKey()) + ",";
					maxCount++;
				} else {
					break;
				}
			}

			if (output.length() > 0)
				output = output.substring(0, output.length() - 1);

			context.write(new Text(key.toString()), new Text(output));
		}
	}
}
