using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
    public class NeuronNetworks
    {

        public Topology Topology { get; }
        public List<Layer> Layers { get; set; }

        private Safekeeping Safekeeping;

        public NeuronNetworks(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            Safekeeping = new Safekeeping();

            CreateInputLayer();
            CreateGiddenLayers();
            CreateOutputLayer();

        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if(Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n=>n.Output).First();    
            }

        }

        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var result = 0.0;
            if (!Safekeeping.CheckNewLayers(Layers, epoch)) {

                foreach (var data in dataset)
                {
                   BackPropagation(data.Item1, data.Item2);
                }
                Layers = Safekeeping.GetLayers(Layers, epoch);
            }
            else
            {
                var error = 0.0;
                for (int i = 0; i < epoch; i++)
                {
                    foreach (var data in dataset)
                    {
                       error += BackPropagation(data.Item1, data.Item2);
                    }
                }
                Safekeeping.Saving(Layers, epoch, this);
                result = error / epoch;
            }

            return result;
        }

        private double BackPropagation(double exprected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - exprected;

            foreach(var neuron in Layers.Last().Neurons)
            { 
                neuron.Learn(difference, Topology.LearningRate);
            }

            for(var j = Layers.Count - 2; j>=0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j+1];

                for(int i = 0; i<layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for(int k = 0; k<previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            var result = Math.Pow(difference, 2);

            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previosLayesSingals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previosLayesSingals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double> { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateGiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayesrs.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayesrs[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

     /*   public void RestoreNeuronConnections(List<List<double>> neuronWeights, List<List<double>> neuronBiases)
        {
            // Проверяем, есть ли достаточно слоев и нейронов для восстановления
            if (neuronWeights.Count != Layers.Count || neuronBiases.Count != Layers.Count)
            {
                throw new ArgumentException("Неверное количество слоев или нейронов для восстановления.");
            }

            // Восстанавливаем веса нейронов и связи между нейронами
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var layerNeuronWeights = neuronWeights[i];
                var layerNeuronBiases = neuronBiases[i];

                // Проверяем, есть ли достаточно весов и смещений для восстановления слоя
                if (layerNeuronWeights.Count != layer.Neurons.Count || layerNeuronBiases.Count != layer.Neurons.Count)
                {
                    throw new ArgumentException("Неверное количество весов или смещений для восстановления слоя.");
                }

                // Восстанавливаем веса нейронов слоя
                for (int j = 0; j < layer.Neurons.Count; j++)
                {
                    var neuron = layer.Neurons[j];
                    var locNeuronWeights = layerNeuronWeights[j];

                    // Проверяем, есть ли достаточно весов для восстановления нейрона
                    if (locNeuronWeights.Count != neuron.Inputs.Count)
                    {
                        throw new ArgumentException("Неверное количество весов для восстановления нейрона.");
                    }

                    // Обновляем веса нейрона
                    neuron.SetWeights(neuronWeights);
                }

                // Восстанавливаем смещения нейронов слоя
                layer.SetBiases(layerNeuronBiases);
            }
        }*/


    }
}
