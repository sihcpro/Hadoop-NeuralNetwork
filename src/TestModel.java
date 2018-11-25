import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.transfer.*;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class TestModel {
	private static int totalLine = 100;
	private static Prob data_test = null;
	private static NeuralNetwork<BackPropagation> ann = new NeuralNetwork<BackPropagation>();

	public static void main(String[] args) {
		generateANN();
		try {
			init("test.csv");
		} catch (Exception e) {
			System.out.println("can't init!");
			e.printStackTrace();
		}
		new TestModel().run();
	}
	
	public static boolean readyData() {
		if( data_test != null )
			return data_test.X.length > 0;
		else
			return false;
	}
	
	public static void init(String file_link) throws Exception {
		if( readyData() )
			return;
		try {
			readFile(file_link);
		} catch (Exception e) {
			System.out.println("Error in init data");
			e.printStackTrace();
			throw e;
		}
		System.out.println("Data size: " + data_test.X.length + " * " + data_test.X[0].length);
	}
	

	private static void generateANN() {
		// TODO Auto-generated method stub
		int inputSize = 6;
		int outputSize = 1;
		int hiddenSize = 10;
		Layer inputLayer = new Layer(inputSize, new NeuronProperties(InputNeuron.class));
		Layer outputLayer = new Layer(outputSize, new NeuronProperties(Neuron.class, TransferFunctionType.SIGMOID));
		Layer hiddenLayer = new Layer(hiddenSize, new NeuronProperties(Neuron.class));
		ann.addLayer(0, inputLayer);
		ann.addLayer(1, hiddenLayer);
		ann.addLayer(2, outputLayer);
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
		ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount() - 1), false);
		ann.setInputNeurons(inputLayer.getNeurons());
		ann.setOutputNeurons(outputLayer.getNeurons());

		TransferFunction[] transarray = {
				new Ramp(),
				new Sgn(), new Sgn(), new Sgn(), new Sgn(), new Sgn(),
				new Sigmoid(),
				new Step(), new Step(),
				new Tanh(),
				};
		double[][] weightHiddenLayer = { { 0.81582, -0.38660, 0.61849, 0.06363, -0.72094, 0.79925 },
				{ -0.93242, 0.88551, -0.38513, 0.76826, 0.29200, 0.32037 },
				{ -0.26446, -0.74611, 0.79431, 0.14814, 0.16109, 0.27192 },
				{ 0.92636, -0.52137, -0.37123, -0.25990, 0.95669, -0.28561 },
				{ 0.64970, -0.44057, 0.51485, 0.62722, 0.68505, 0.72473 },
				{ 0.41644, -1.08351, -0.20779, 0.63705, -0.08367, -1.12827 },
				{ -0.01389, -0.93703, -0.02113, -0.33359, 0.31248, -0.74139 },
				{ -0.91919, -0.62536, -0.03147, -0.96627, 0.96529, -1.06998 },
				{ 0.27612, -0.69984, 0.16853, -0.22499, -0.66187, -0.87747 },
				{ 0.11135, -0.04549, -0.80816, 0.78771, 0.82315, 0.03679 } };
		double[] weightOutputLayer = { -0.63077, 0.46545, -0.84510, 0.49579, -0.39947, -1.11113, -0.92727, -0.48606, -0.79375, 0.75722,
				0.00191, -0.00467, -0.45812, -0.36849, 1.03047, 0.72785 };
		
		for (int i = 0; i < hiddenSize; i++) {
			Neuron current = hiddenLayer.getNeuronAt(i);
			current.setTransferFunction(transarray[i]);
			for (int j = 0; j < inputSize; j++) {
				current.getConnectionFrom(inputLayer.getNeuronAt(j)).setWeight(new Weight(weightHiddenLayer[i][j]));
			}
		}
		Neuron current = outputLayer.getNeuronAt(0);
		for (int i = 0; i < hiddenSize; i++) {
			current.getConnectionFrom(hiddenLayer.getNeuronAt(i)).setWeight(new Weight(weightOutputLayer[i]));
		}
	}

	private void run() {
		System.out.println("Start testing");
		boolean result[] = new boolean[data_test.X.length];
		double threshold = 0.5;
		int count = 0;
		for (int i = 0; i < totalLine; i++) {
			double x = data_test.X[i][0] / 10.0;
			double y = data_test.X[i][1] / 1000.0;
			double[] tinp = { x, y, x * x, y * y, Math.sin(x), Math.sin(y) };
			ann.setInput(tinp);
			ann.calculate();

			int answer = ann.getOutput()[0] > threshold ? 1 : 0;
			System.out.print("Result: " + data_test.y[i] + " vs " + answer + " (" + ann.getOutput()[0] + ")\n");
			result[i] = answer == data_test.y[i];
			if (result[i])
				count++;
		}
		System.out.println("Finished testing");
		System.out.println("Result: Test with " + result.length + " cases | correct in " + count + " cases | ratio " + ((float) count / result.length));
		
	}

	/*
	 */
	public static void readFile(String path) throws Exception {
		BufferedReader reader = null;
		List<List<Double>> tmp_data = new ArrayList<List<Double>>();
		List<String> name = new ArrayList<String>();

		try {
			reader = new BufferedReader(new FileReader(new File(path)));
			String text = null;

			int count = -1;
			while ((text = reader.readLine()) != null) {
				if (count == -1) {
					name.add(text);
					count++;
				} else {
					String[] tmp = text.split(",");
					tmp_data.add(new ArrayList<Double>());
					for (String i : tmp) {
						tmp_data.get(count).add(Double.parseDouble(i));
					}
					count++;
				}
			}
			totalLine = tmp_data.size();

			data_test = Neural.changeTypeProblem(tmp_data);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (reader != null) {
					reader.close();
				}
			} catch (IOException e) {
			}
		}
	}
}
