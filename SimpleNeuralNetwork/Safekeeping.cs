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
        public async void Saving(List<Layer> Layers, NeuronNetworks neuronNetworks)
        {
            if (!this.checkNewLayers(Layers))
            {
                string layersStr = this.getThisLayers(Layers);
                return;
            }

            int countLayersHiddent = Layers.Count - 1;
            List<List<double>> neuron;

            string key = this.calculationKeyByLayers(Layers);

            string info = "";
            for(int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].Type == NeuronType.Normal)
                {
                    info += this.calculationInfoByLayer(Layers[i]);
                    info += "/";
                }
            }


           

            //https://metanit.com/sharp/tutorial/5.5.php
            using (StreamWriter writer = new StreamWriter(AllPath, true, System.Text.Encoding.Default))
                {
                   await writer.WriteLineAsync(key + info);
                //await writer.WriteAsync(info); 
               // await writer.WriteLineAsync("####");
                //await writer.WriteAsync("####");
            }


        }


        private string calculationKeyByLayers(List<Layer> Layers)
        {
            var neuronsCount = Layers
                .Where(layer => layer.Type == NeuronType.Normal)
                .Select(layer => layer.Neurons.Count.ToString());

            return string.Join(":", neuronsCount);
        }

        private string calculationInfoByLayer(Layer Layer)
        {
            string info = "#";
            foreach (var neuron in Layer.Neurons)
            {
                info += neuron.Delta.ToString();
                info += '{';
                info += string.Join(":", neuron.Weights);
                info += '}';
            }

            return info;
        }

        private bool checkNewLayers(List<Layer> Layers)
        {
            if (!File.Exists(AllPath) || File.ReadAllLines(AllPath).Length == 0)
                return true;

            string thisKey = this.calculationKeyByLayers(Layers);

            foreach (string line in File.ReadAllLines(AllPath))
                if (thisKey == line.Split('#')[0])
                    return false;

            return true;
        }

        private string getThisLayers(List<Layer> Layers)
        {
            if (!File.Exists(AllPath) || File.ReadAllLines(AllPath).Length == 0)
                return "";

            string thisKey = this.calculationKeyByLayers(Layers);

            foreach (string line in File.ReadAllLines(AllPath))
                if (thisKey == line.Split('#')[0])
                    return line.Split('#')[1];

            return "";
        }

    }
}
