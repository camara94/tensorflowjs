const run = async() => {
    const csvURL = './../data/titanic.csv';

    const data = [
        { index: 0, value: 50 },
        { index: 1, value: 100 },
        { index: 2, value: 150 },
      ];
      
      // Get a surface
      const surface = tfvis.visor().surface({ name: 'Barchart', tab: 'Charts' });
      
      // Render a barchart on that surface
      tfvis.render.barchart(surface, data, {});

    const trainingData = tf.data.csv(csvURL, {
        columnConfigs: {
            Survived: {
                isLabel: true
            }
        }
    });

    const numberOfFeatures = ((await trainingData.columnNames()).length-1);
    
    const convertedData = trainingData.map(({xs, ys})=>{
        return { xs: Object.values(xs), ys: Object.values(ys) };
    }).batch(10);

    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units:4,
            inputShape: numberOfFeatures,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 2,
            activation: 'softmax'
        })
    );
    
    model.summary();
    model.compile({
        optimizer: tf.train.adam(0.06),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    await model.fitDataset(convertedData, {
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: async(epoch, logs) => {
                console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss + ' Accuracy: ' + logs.acc)
            }
        }
    });
    const labels = ['Mort', 'Survecu'];
    const sample = tf.tensor2d([3,1,34.5], [1, 3]);
    const prediction = model.predict(sample);
    const indexMax = tf.argMax(prediction, axis=1).dataSync();
    alert(labels[indexMax])
}

run();