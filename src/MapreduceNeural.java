import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

//hdfs namenode -format

public class MapreduceNeural {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Result>{

	public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		FileSystem fs = FileSystem.get(conf);
		String file_link, itr[] = value.toString().split("\n");
		int total_line = 20, maxIterations = 10, loop = 1;
		double ratio = 0.9;
		try {
			file_link = conf.get("file_link");
			total_line = conf.getInt("total_line", 100);
			ratio  = conf.getFloat("ratio", (float) 0.9) % 1.0;
			maxIterations = conf.getInt("maxIterations", 10);
			loop = conf.getInt("loop", 1);
		} catch (Exception e) {
			Result r = new Result(0, "Error"+e.getMessage());
			context.write(new Text("Error"), r);
			return;
		}
		Neural cluster = new Neural(total_line, ratio, maxIterations, loop, fs);;
		try {
			Neural.init(file_link);
		} catch (Exception e) {
		}
		for ( String i : itr) {
			try {
				String[] tmp = i.split(",");
				int kindNeural[] = new int[tmp.length];
				for(int j = 0; j < tmp.length; j++) {
					kindNeural[j] = Integer.parseInt(tmp[j]);
				}
				Result answer = cluster.run(kindNeural);
				context.write(new Text(i), answer);
			} catch (Exception e) {
				Result r = new Result(0, " !!! "+e.getMessage());
				context.write(new Text(i), r);
			}
		}
	}
		
		public List< List<Integer> > transferToData(String data) {
			String a[] = data.toString().split("],");
			List< List<Integer> > data2 = new ArrayList< List<Integer> >();
			for(String s : a) {
				s = s.replace("[", "");
				s = s.replace("]", "");
				String s2[] = s.split(",");
				List<Integer> list = new ArrayList<Integer>();
				for(String i : s2) {
					list.add(Integer.parseInt(i));
				}
				data2.add(list);
			}
			return data2;
		}
	}

	public static class NeuralReducer extends Reducer<Text,Result,Text,Result> {
		private static Result result = new Result(-1, "");

		public void reduce(Text key, Iterable<Result> values, Context context) throws IOException, InterruptedException {
			for (Result val : values) {
				if( val.point > result.point ) {
					result = new Result(val.point, "\n" + key.toString() + "\n" + val.result);
				}
			}
			context.write(new Text("result"), result);
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		BufferedReader in;
		String inputFile = "/inputNeural", outputFile = "/output";
		try {
			Path input = new Path(args[0]);
			if( fs.exists(input) ) {
				inputFile = args[0];
				System.out.println("File input set : " + inputFile);
			} else if(fs.exists(new Path("/inputNeural"))) {
				inputFile = "/inputNeural";
				System.out.println("File input default : " + inputFile);
			}
		} catch (Exception e) {
			if(fs.exists(new Path("/inputNeural"))) {
				inputFile = "/inputNeural";
				System.out.println("File input default : " + inputFile);
			} else {
				throw new Error("Wrong file input");
			}
		}
		try {
			Path output = new Path(args[1]);
			if ( !fs.exists(output) ) {
				outputFile = args[1];
				System.out.println("File output set : " + outputFile);
			} else if ( !fs.exists(new Path("/output")) ) {
				outputFile = "/output";
				System.out.println("File output default : " + outputFile);
			} else if ( !fs.exists(new Path("/output" + args[args.length-1])) ) {
				outputFile = "/output" + args[args.length-1];
				System.out.println("File output default : " + outputFile);
			}
		} catch (Exception e) {
			try {
				if( !fs.exists(new Path("/output")) ) {
					outputFile = "/output";
					System.out.println("File output default : " + outputFile);
				} else if ( !fs.exists(new Path("/output" + args[args.length-1])) ) {
					outputFile = "/output" + args[args.length-1];
					System.out.println("File output default : " + outputFile);
				}
			} catch (Exception e2) {
				boolean delete = false;
				try {
					fs.delete(new Path("/output/_SUCCESS"), delete);
				} catch (Exception e3) {
				}
				try {
					fs.delete(new Path("/output/part-r-00000"), delete);
				} catch (Exception e3) {
				}
				try {
					fs.delete(new Path("/output"), delete);
				} catch (Exception e3) {
					e3.printStackTrace();
				}
				if (!delete) {
					System.out.println("Path /output had been deleted");
					System.out.println("File output default : " + outputFile);
					outputFile = "/output";
				} else {
					throw new Error("Wrong output file!");
				}
			}
		}
		try {
			in = new BufferedReader(new InputStreamReader(fs.open(new Path(args[2]))));
			if( in.ready() && in.readLine().length() > 0 ) {
				conf.set("file_link", args[2]);
			}
			in.close();
			System.out.println("File train set : " + args[2]);
		} catch (Exception e) {
			conf.set("file_link", "/train.csv");
			System.out.println("File train default : /train.csv");
		}
		try {
			int total_line = Integer.parseInt(args[3]);
			conf.setInt("total_line", total_line);
			System.out.println("Total line set : " + total_line);
		} catch (Exception e) {
			conf.setInt("total_line", 900);
			System.out.println("Total line default : 100");
		}
		try {
			float ratio = Float.parseFloat(args[4]);
			conf.setFloat("ratio", ratio);
			System.out.println("Ratio set : " + ratio);
		} catch (Exception e) {
			conf.setFloat("ratio", (float) 0.9);
			System.out.println("Ratio default : 0.9");
		}
		try {
			int maxIterations = Integer.parseInt(args[5]);
			conf.setInt("maxIterations",  maxIterations);
			System.out.println("Max Iterationsset set : " + args[5]);
		} catch (Exception e) {
			conf.setInt("maxIterations", 20);
			System.out.println("Max Iterationsset default : 10");
		}
		try {
			int loop = Integer.parseInt(args[6]);
			conf.setInt("loop",  loop);
			System.out.println("Loop set : " + args[6]);
		} catch (Exception e) {
			conf.setInt("loop", 1);
			System.out.println("Loop default : 1");
		}
		
		Job job = Job.getInstance(conf, "Train neural");
		job.setJarByClass(MapreduceNeural.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(NeuralReducer.class);
		job.setReducerClass(NeuralReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Result.class);
		FileInputFormat.addInputPath(job, new Path(inputFile));
		FileOutputFormat.setOutputPath(job, new Path(outputFile));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}


}

