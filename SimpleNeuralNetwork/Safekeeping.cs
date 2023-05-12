using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
    internal class Safekeeping
    {
        private const string path = "";
        private string NameFile = "testKeepingInTouch";
        private const string TypeFile = ".txt";
        private string AllPath;



        public Safekeeping()
        {
            AllPath = path + NameFile + TypeFile;
        }
        public async void Saving(List<Layer> Layers, int epoch, NeuronNetworks neuronNetworks)
        {
            if (!this.CheckNewLayers(Layers, epoch))
            {
                string layersStr = this.GetThisLayers(Layers, epoch);

                return;
            }

            int countLayersHiddent = Layers.Count - 1;
            List<List<double>> neuron;

            string key = this.CalculationKeyByLayers(Layers, epoch);

            string info = "#";
            for(int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].Type == NeuronType.Normal)
                {
                    info += this.CalculationInfoByLayer(Layers[i]);
                    info += "/";
                }
            }

            //https://metanit.com/sharp/tutorial/5.5.php
            using (StreamWriter writer = new StreamWriter(AllPath, true, System.Text.Encoding.Default))
                {
                   await writer.WriteLineAsync(key + info);
            }


        }


        private string CalculationKeyByLayers(List<Layer> Layers, int epoch)
        {
            var neuronsCount = Layers
                .Where(layer => layer.Type == NeuronType.Normal)
                .Select(layer => layer.Neurons.Count.ToString());

            return string.Join(":", neuronsCount) + ':' + epoch.ToString();
        }

        public bool CheckNewLayers(List<Layer> Layers, int epoch)
        {
            if (!File.Exists(AllPath) || File.ReadAllLines(AllPath).Length == 0)
                return true;

            string thisKey = this.CalculationKeyByLayers(Layers, epoch);

            foreach (string line in File.ReadAllLines(AllPath))
                if (thisKey == line.Split('#')[0])
                    return false;

            return true;
        }

        private string GetThisLayers(List<Layer> Layers, int epoch)
        {
            if (!File.Exists(AllPath) || File.ReadAllLines(AllPath).Length == 0)
                return "";

            string thisKey = this.CalculationKeyByLayers(Layers, epoch);

            foreach (string line in File.ReadAllLines(AllPath))
                if (thisKey == line.Split('#')[0])
                    return line.Split('#')[1];

            return "";
        }

        private string CalculationInfoByLayer(Layer Layer)
        {
            string info = "";
            foreach (var neuron in Layer.Neurons)
            {
                info += neuron.Delta.ToString();
                info += '{';
                info += string.Join(":", neuron.Weights);
                info += '}';
            }

            return info;
        }
        public List<Layer> GetLayers(List<Layer> layers, int epoch)
        {
            string thisLayersString = GetThisLayers(layers, epoch);
            string[] arrayLayers = thisLayersString.Split('/');
            int layerIndex = 0;

            foreach (var layer in layers)
            {
                if (layer.Type == NeuronType.Normal)
                {
                    string layerString = arrayLayers[layerIndex];
                    string[] neuronsArray = layerString.Split('}');
                    int neuronIndex = 0;

                    foreach (var neuron in layer.Neurons)
                    {
                        string neuronString = neuronsArray[neuronIndex];

                        double delta = double.Parse(neuronString.Split('{')[0]);
                        neuron.SetDelta(delta);

                        string weightsString = neuronString.Split('{')[1];
                        string[] weights = weightsString.Split(':');
                        List<double> weightList = weights.Select(w => double.Parse(w)).ToList();
                        neuron.SetWeights(weightList);

                        neuronIndex++;
                    }

                    layerIndex++;
                }
            }

            return layers;
        }


    }
}
