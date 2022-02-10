const run = async () => {
    // url du dataset
    const csvURL = './../data/titanic.csv';

    // chargement du dataset
    const trainingData = tf.data.csv(csvURL, {
        columnConfigs: {
            Survived: {
                isLabel: true
            }
        }
    });

    // récupérer le nombres de feature ou variables indépendantes
    const numbersOfFeatures = (await trainingData.columnNames()).length - 1;
    
    // tranformer les données
    const convertedData = trainingData.map(({xs, ys}) => {
        return { xs: Object.values(xs), ys: Object.values(ys) }
    }).batch(10);

    // création du model
    const model = tf.sequential()

    model.add( tf.layers.dense( { 
        units: 4, 
        inputShape: [ numbersOfFeatures ], 
        activation: 'relu' } ) 
    );
    model.add( tf.layers.dense( { units: 2, activation: 'sigmoid' } ) );
    model.summary()

    // compilation du modèle
    model.compile( { 
        optimizer: tf.train.adam(0.06), 
        loss: 'meanSquaredError',
        metrics: ['accuracy'] 
    } );

    // entrainement du modèle
    await model.fitDataset( convertedData, {
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(
                    'Epoch: ' + epoch 
                  + ' Loss: ' + logs.loss
                  + ' Accuracy: ' + logs.acc 
                  )
            }
        }
    } );

    // prédictions
    const columnNames = ['Mort', 'Survécu'];
    const feature = tf.tensor2d([3,0,47.0], [1, 3]);
    let prediction = model.predict(feature);
    const indexMax = tf.argMax(prediction, axis=1).dataSync();
    alert(columnNames[indexMax]);
    
}

run()