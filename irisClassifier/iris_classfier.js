const run = async () => {
    // url du fichier csv
    const csvURL = './../data/iris.csv';
    
    // chargement du fichier
    const trainData = tf.data.csv(csvURL, {
        columnConfigs: {
            species: {
                isLabel: true
            }
        }
    });
    
    // récuperer le nombre de variable indépendante
    const numberOfFeatures = ((await trainData.columnNames()).length-1)
    // le nombre de ligne dans la base
    const numberOfSamples = 150;
    
    // transformation de la base avec one-hot encoding
    const convertedData = trainData.map(({xs, ys}) => {
        const labels = [
            ys.species == 'setosa' ? 1 : 0,
            ys.species == 'virginica' ? 1 : 0,
            ys.species == 'versicolor' ? 1 : 0
        ];
        return { xs: Object.values(xs), ys: Object.values(labels) };
    }).batch(50);

    // création du modèle
    const model = tf.sequential()
    model.add( tf.layers.dense({  
                                inputShape: [numberOfFeatures], 
                                activation: "sigmoid",
                                units: 5 
                            }));

    model.add( tf.layers.dense({
        activation: 'softmax',
        units: 3
    }) );

    //model.summary();
    model.compile({loss: "categoricalCrossentropy", optimizer: tf.train.adam(0.06)});
    // en trainement du modèle
    await model.fitDataset( convertedData, {
        epochs: 100,
        callbacks: {
            onEpochEnd: async(epoch, logs) => {
                console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss );
            }
        }});
         
    // Setosa
    const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
    const prediction = model.predict(testVal);
    const pIndex = tf.argMax(prediction, axis=1).dataSync();
    const classNames = ["Setosa", "Virginica", "Versicolor"];
            
    alert(classNames[pIndex]);  
}
run();