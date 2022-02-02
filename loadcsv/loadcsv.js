const csvURL = './../data/iris.csv';

const trainData = tf.data.csv(csvURL, {
    columnConfigs: {
        species: {
            isLabel: true
        }
    }
});

console.log(trainData);