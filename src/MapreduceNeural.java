import java.io.*;
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
//import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.transfer.*;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;


//hdfs namenode -format

public class MapreduceNeural {
	private static List< List<Integer> > data = new ArrayList< List<Integer> >();
	private static List< List<Integer> > test = new ArrayList< List<Integer> >();
	private static List<String> name = new ArrayList<String>();
	private static Neural neural;
	private static int lengthOutputLayer = 8;
	private static int inputSize = 784;
	private static int outputSize = 8;
	public static DataSet ds = new DataSet(inputSize, outputSize);

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text>{

//    private final static IntWritable one = new IntWritable(1);
//		private Text word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			String file_link, itr[] = value.toString().split("\n");
//			Prob data_train, data_test;
			int limit_train = 20, limit_test = 10, maxIterations = 10;
			try {
				file_link = conf.get("file_link");
				limit_train = conf.getInt("limit_train", 20);
				limit_test  = conf.getInt("limit_test", 10);
				maxIterations = conf.getInt("maxIterations", 10);
//				data_train = Neural.changeTypeProblem(transferToData(conf.get("data")));
//				data_test  = Neural.changeTypeProblem(transferToData(conf.get("test")));
			} catch (Exception e) {
				context.write(new Text("Error"), new Text(e.getMessage()));
				return;
			}
			try {
				Neural.readfile(fs, file_link, limit_train, limit_test);
//				context.write(new Text("read file"), new Text("done"));
			} catch (Exception e) {
//				context.write(new Text("read file"), new Text("error " + e.getMessage()));
			}
			for ( String i : itr) {
				try {
					Neural cluster = new Neural();
					String[] tmp = i.split(",");
					int kindNeural[] = new int[tmp.length];
					for(int j = 0; j < tmp.length; j++) {
						kindNeural[j] = Integer.parseInt(tmp[j]);
					}
					String answer = cluster.run(kindNeural, maxIterations);
					context.write(new Text(i), new Text(answer));
				} catch (Exception e) {
					context.write(new Text(i), new Text(" !!! "+e.getMessage()));
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

	public static class IntSumReducer extends Reducer<Text,Text,Text,Text> {
		private Text result = new Text();

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			String sum = "";
			for (Text val : values) {
				sum += val.toString();
			}
			result.set(new Text(sum));
			context.write(key, result);
		}
	}

	public static void main(String[] args) throws Exception {
		neural = new Neural();
		Neural.init();
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		BufferedReader in;
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
			int limit_train = Integer.parseInt(args[3]);
			int limit_test  = Integer.parseInt(args[4]);
			conf.setInt("limit_train", limit_train);
			conf.setInt("limit_test",  limit_test);
			System.out.println("Train leng and test leng set : " + limit_train + "/" + limit_test);
		} catch (Exception e) {
			conf.setInt("limit_train", 20);
			conf.setInt("limit_test",  10);
			System.out.println("Train leng and test leng default : 200/100");
		}
		try {
			int maxIterations = Integer.parseInt(args[5]);
			conf.setInt("maxIterations",  maxIterations);
			System.out.println("Max Iterationsset set : " + args[5]);
		} catch (Exception e) {
			conf.setInt("maxIterations", 10);
			System.out.println("Max Iterationsset default : 10");
		}

//		conf.set("data", data.toString().replace(" ", ""));
//		conf.set("test", test.toString().replace(" ", ""));
		
		System.out.println("file link     : " + conf.get("file_link"));
		System.out.println("limit train   : " + conf.getInt("limit_train", 22));
		System.out.println("limit test    : " + conf.getInt("limit_test", 11));
		System.out.println("maxIterations : " + conf.getInt("maxIterations", 11));
		
		Job job = Job.getInstance(conf, "Train neural");
		job.setJarByClass(MapreduceNeural.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setCombinerClass(IntSumReducer.class);
		job.setReducerClass(IntSumReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}


	public static void readfile(FileSystem fs, String file_link, int limit, int test_leng) throws Exception{
		int limit_read = limit + test_leng;
		BufferedReader in = null;
		
		try {
			in = new BufferedReader(new InputStreamReader(fs.open(new Path(file_link))));
		    String text = null;
		    
		    int count = -1;
		    while ((text = in.readLine()) != null && count < limit_read) {
		    	if(count == -1) {
		    		name.add(text);
		    		count++;
		    	} else {
		    		String[] tmp = text.split(",");
		    		data.add(new ArrayList<Integer>());
		    		for(String i : tmp) {
		    			data.get(count).add(Integer.parseInt(i));
		    		}
		    		count++;
		    	}
		    }

			for(int i = 0; i < test_leng; i++) {
				test.add(data.get(data.size()-1));
				data.remove(data.size()-1);
			}
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		} catch (IOException e) {
		    e.printStackTrace();
		} finally {
		    try {
		        if (in != null) {
		            in.close();
		        }
		    } catch (IOException e) {
		    }
		}
	}
}







//--------------------------------------------------------------------------




class Neural {
	private static List< List<Integer> > data = new ArrayList< List<Integer> >();
	private static List< List<Integer> > test = new ArrayList< List<Integer> >();
	private static List<String> name = new ArrayList<String>();
	private static Prob data_train;
	private static Prob data_test;
	
	Neural(){
		init();
	}
	
	public static double[][] truthValue = new double[][] {
//		{1,0,0,0,0,0,1,0,1,1}, // 0
//		{0,1,0,0,0,0,0,0,0,0},
//		{0,0,0,1,0,0,0,1,1,0},
//		{0,0,0,0,1,0,0,1,1,0},
//		{0,1,0,1,0,0,0,0,0,0},
//		{0,0,1,0,1,1,1,1,1,0}, // 5
//		{1,0,0,0,0,0,1,1,0,1},
//		{0,1,0,1,0,0,0,0,1,0},
//		{1,0,1,0,0,1,1,1,1,1},
//		{1,0,1,0,1,0,1,0,1,0}, // 9

	
		{1,0,0,0,0,0,1,1}, // 0
		{0,1,0,0,0,0,0,0},
		{0,0,0,1,0,0,1,0},
		{0,0,0,0,1,0,1,0},
		{0,1,0,1,0,0,0,0},
		{0,0,1,0,1,1,1,0}, // 5
		{1,0,0,0,0,0,0,1},
		{0,1,0,1,0,0,1,0},
		{1,0,1,0,0,1,1,1},
		{1,0,1,0,1,0,1,0}, // 9
	};
	
	public static Map<String, Integer> value = new HashMap<String, Integer>();

	public static void init() {
		for(int i = 0; i < 10; i++) {
			value.put(toString(truthValue[i]), i+1);
		}
	}
	
	private static String toString(double[] value) {
		String tmp = "";
		for(double i : value) {
			tmp = tmp + (int)i;
		}
		return tmp;
	}
	
	private static Neuron chooseNeural(int i) {
		Neuron n = new Neuron();
		// Linear, Log, Ramp, Sgn, Sigmoid, Sin, Step, Tanh, Trapezoid
		switch( i%6 ) {
		case 0:
			// 200 0.15
			n.setTransferFunction(new Linear(0.26));
			break;
		case 1:
			// 200 0.3
			n.setTransferFunction(new Ramp());
			break;
		case 2:
			// 200 0
			n.setTransferFunction(new Sgn());
			break;
		case 3:
			// 200 0.20
			n.setTransferFunction(new Sigmoid(1.0));
			break;
		case 4:
			// 200 0.35
			n.setTransferFunction(new Step());
			break;
		case 5:
			// 200 0.1
			n.setTransferFunction(new Tanh());
			break;
		default:
			// 200 0.3
		}
		return n;
	}
	
//	public static void main(String[] args) {
//		run(new int[] {0,0,0,0,0,0,0,0,0});
//	}
	
	public String run(int[] kindNeuron, int maxIterations) {
		String output = "";
		init();
		
		int inputSize = 784;
		int hiddenSize = 40;
		int outputSize = 10;
		DataSet ds = new DataSet(inputSize, outputSize);

		int maxLen = kindNeuron.length;
		Layer inputLayer = new Layer();
		Layer hiddenLayerOne = new Layer();
		Layer outputLayer = new Layer();
		
		if( maxLen == 0 ) {
			kindNeuron = new int[] {0};
		}
		for(int i = 0; i < inputSize; i++) {
			inputLayer.addNeuron(new Neuron());
		}
		int amountNeuronHidden = 0;
		for(int i = 0; i < 5; i++) {
			for(int j = 0; j < kindNeuron[j%maxLen]; j++) {
				if( amountNeuronHidden >= 40 )
					continue;
				hiddenLayerOne.addNeuron(chooseNeural(i));
				amountNeuronHidden++;
			}
		}
		for(; amountNeuronHidden < hiddenSize; amountNeuronHidden++) {
			hiddenLayerOne.addNeuron(chooseNeural(5));
			amountNeuronHidden++;
		}
		for(int i = 0; i < outputSize; i++) {
			Neuron n = new Neuron();
			n.setTransferFunction(new Sigmoid(1.0));
			outputLayer.addNeuron(n);
		}

		try {
//			readfile(fs, "/train.csv", 3000, 500);
//			Prob data_train = changeTypeProblem(data);
			System.out.println("Data_train size: "+data_train.X.length+" * "+data_train.X[0].length);
			System.out.println("Add data set");
			double data_train_y[] = new double[outputSize];
			for(int i = 0; i < outputSize; i++) {
				data_train_y[i] = 0;
			}
			int value = 0;
			for(int i = 0; i < data_train.X.length; i++) {
				value = (int)data_train.y[i];
				data_train_y[ value ] = 1;
				ds.addRow(data_train.X[i], data_train_y);
				data_train_y[ value ] = 0;
			}
		} catch (Exception e) {
			System.out.println("Error in Adding");
			e.printStackTrace();
		}
		
		output += "-> train leng " + data_train.X.length + " & test leng " + data_test.X.length;

		NeuralNetwork ann;
		try {
			System.out.println("Training data");
			ann = new NeuralNetwork();
			ann.addLayer(0, inputLayer);
			ann.addLayer(1, hiddenLayerOne);
			ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
			ann.addLayer(2, outputLayer);
			ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
			ConnectionFactory.fullConnect(ann.getLayerAt(0), 
			ann.getLayerAt(ann.getLayersCount()-1), false);
			ann.setInputNeurons(inputLayer.getNeurons());
			ann.setOutputNeurons(outputLayer.getNeurons());
			BackPropagation backPropagation = new BackPropagation();
			backPropagation.setMaxIterations(maxIterations);
			long startTime = System.nanoTime();
			ann.learn(ds, backPropagation);
			ds = null;
			long endTime = System.nanoTime();
			System.out.println("Running time = "+(endTime - startTime)/1000000000+" s");
		} catch (Exception e) {
			return e.getMessage();
		}
		
		output += " -> trained";
		
		try {
			System.out.println("Trainfer Test");
//			Prob data_test = changeTypeProblem(test);
			System.out.println("Start get results");
			boolean result[] = new boolean[data_test.X.length];
			boolean result2[] = new boolean[data_test.X.length];
			int count = 0, count2 = 0;
			for(int i = 0; i < data_test.X.length; i++) {
				ann.setInput(data_test.X[i]);
				ann.calculate();
				double[] networkOutputOne = ann.getOutput();
				
				System.out.print("Result: "+data_test.y[i]+"      vs      ");
				if( !value.containsKey(toString(networkOutputOne)) ) {
					System.out.print("non  |  ");
					System.out.print(toString(truthValue[(int)data_test.y[i]])+" vs ");
					System.out.print(toString(networkOutputOne));
				} else {
					double answer = value.get(toString(networkOutputOne)) - 1;
					System.out.print(answer);
					result[i] = data_test.y[i] == answer;
					if( result[i] ) count++;
				}
				result2[i]= data_test.y[i] == nearest(networkOutputOne);
				if( result2[i] ) count2++;
				System.out.println();
			}
			System.out.println("Done");
			System.out.println(result);
			System.out.println("Result: "+((float)count / result.length));
			return output + " -> " + ((float)count / result.length) + " -> " + ((float)count2 / result.length);
		} catch (Exception e) {
			System.out.println("Error in get result");
			e.printStackTrace();
			return e.getMessage();
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	private static int nearest(double []result) {
		int num = 0, count = 0, bestmatch = 0;
		double x, y;
		try {
			for(int i = 0; i < 10; i++) {
				count = result.length;
				for(int j = 0; j < result.length; j++) {
					x = result[j];
					y = Neural.truthValue[i][j];
					if(x * y == 0 && x + y != 0) {
						count--;
					}
				}
				if( count > bestmatch ) {
					bestmatch = count;
					num = i;
				}
			}
			return num;
		} catch (Exception e) {
			return num;
		}
	}
	
	public static void readfile(FileSystem fs, String file_link, int limit, int test_leng) throws Exception{
		int limit_read = limit + test_leng;
		BufferedReader in = null;
		
		try {
			in = new BufferedReader(new InputStreamReader(fs.open(new Path(file_link))));
		    String text = null;
		    
		    int count = -1;
		    while ((text = in.readLine()) != null && count < limit_read) {
		    	if(count == -1) {
		    		name.add(text);
		    		count++;
		    	} else {
		    		String[] tmp = text.split(",");
		    		data.add(new ArrayList<Integer>());
		    		for(String i : tmp) {
		    			data.get(count).add(Integer.parseInt(i));
		    		}
		    		count++;
		    	}
		    }

			for(int i = 0; i < test_leng; i++) {
				test.add(data.get(data.size()-1));
				data.remove(data.size()-1);
			}
			
			data_train = changeTypeProblem(data);
			data_test  = changeTypeProblem(test);
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		} catch (IOException e) {
		    e.printStackTrace();
		} finally {
		    try {
		        if (in != null) {
		            in.close();
		        }
		    } catch (IOException e) {
		    }
		}
	}
	
	public static Prob changeTypeProblem(List< List<Integer> > data) throws Exception{
		Prob prob = new Prob();

		try {
			if(data == null || data.size() == 0) {
				System.out.println("Error data is empty!");
				return new Prob();
			} else if(data.get(0).size() == 0){
				System.out.println("Error data[0] is empty!");
				return new Prob();
			}
			
			int leng = data.get(0).size() - 1;
			prob.X = new double[data.size()][leng];
			prob.y = new double[data.size()];
			
			for(int i = 0; i < data.size(); i++) {
				prob.y[i] = data.get(i).get(0);

				for(int j = 0; j < leng; j++) {
					prob.X[i][j] = data.get(i).get(j+1);
				}
			}
		} catch (Exception e) {
			System.out.println("Error in make Problem");
			System.out.println(e.toString());
			e.printStackTrace();
			throw e;
		}
		
		return prob;
	}
	
}

class Prob {
	public String[] name;
	public double[][] X;
	public double[] y;
	
//	Prob Prob(int lengX, int lengY) {
//		this.name = new String[lengY];
//		this.X = new float[lengX][lengY];
//		this.y = new float[lengX];
//		return this;
//	}
}

