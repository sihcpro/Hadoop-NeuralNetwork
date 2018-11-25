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
import org.neuroph.core.transfer.Linear;
import org.neuroph.core.transfer.Ramp;
import org.neuroph.core.transfer.Sgn;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.core.transfer.TransferFunction;
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

		TransferFunction[] transarray = { new Linear(), new Linear(), new Ramp(), new Ramp(), new Ramp(), new Ramp(),
				new Sgn(), new Sgn(), new Sgn(), new Sigmoid() };
		double[][] weightHiddenLayer = { { -0.21416, -1.70727, 1.11227, 0.57532, -0.46455, -0.59969 },
				{ -0.21416, -1.70727, 1.11227, 0.57532, -0.46455, -0.59969 },
				{ -0.08718, 2.33875, -3.40992, -1.18772, 0.33032, 2.36377 },
				{ -0.02908, -0.56299, 2.15620, 0.40209, -0.82653, -0.71587 },
				{ -0.91916, -0.24767, 1.60806, -0.43138, -0.01356, -0.46046 },
				{ 1.10098, -0.12995, 0.78374, -0.61391, 0.70339, 0.53580 },
				{ -0.13611, 0.21901, -0.31842, -0.34229, -0.85729, 0.95568 },
				{ 0.96965, -1.98639, 2.07623, -0.32663, -0.96285, -0.53443 },
				{ -0.46673, -0.53107, 1.09887, -0.29202, -0.85990, -1.03142 },
				{ 0.65444, 2.41845, -1.44543, 0.19785, 0.31301, 2.20465 },
				{ 0.77626, -1.01033, 0.30118, -0.63921, 0.20845, -0.45220 } };
		double[] weightOutputLayer = { -2.26483, 4.51318, -1.78639, -0.55019, 0.95590, 0.62558, -1.19854, -1.86492,
				1.52334, 0.35091, -0.92233, 1.55792, -1.07835, 0.00218, -0.01617, 1.73879 };
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
